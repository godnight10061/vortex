// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::ToCanonical;
use vortex_array::arrays::ListArray;
use vortex_array::compute::filter;
use vortex_array::compute::sub_scalar;
use vortex_array::expr::Expression;
use vortex_array::expr::Root;
use vortex_array::expr::root;
use vortex_array::validity::Validity;
use vortex_dtype::DType;
use vortex_dtype::FieldMask;
use vortex_dtype::Nullability;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::ArrayFuture;
use crate::LayoutReader;
use crate::LazyReaderChildren;
use crate::layouts::USE_VORTEX_OPERATORS;
use crate::layouts::list::ListLayout;
use crate::segments::SegmentSource;

pub struct ListReader {
    layout: ListLayout,
    name: Arc<str>,
    lazy_children: LazyReaderChildren,
}

impl ListReader {
    pub(super) fn try_new(
        layout: ListLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: VortexSession,
    ) -> VortexResult<Self> {
        let is_nullable = layout.dtype().is_nullable();
        let mut dtypes = Vec::new();
        let mut names = Vec::new();

        if is_nullable {
            dtypes.push(DType::Bool(Nullability::NonNullable));
            names.push(Arc::from("validity"));
        }

        // Offsets
        dtypes.push(layout.offsets_dtype().clone());
        names.push(Arc::from("offsets"));

        // Elements
        dtypes.push(
            layout
                .dtype()
                .as_list_element_opt()
                .vortex_expect("list element")
                .as_ref()
                .clone(),
        );
        names.push(Arc::from("elements"));

        let lazy_children = LazyReaderChildren::new(
            layout.children().clone(),
            dtypes,
            names,
            segment_source,
            session,
        );

        Ok(Self {
            layout,
            name,
            lazy_children,
        })
    }
}

impl LayoutReader for ListReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    fn register_splits(
        &self,
        _field_mask: &[FieldMask],
        row_range: &Range<u64>,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        splits.insert(row_range.end);
        Ok(())
    }

    fn pruning_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &Expression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        Ok(MaskFuture::ready(mask))
    }

    fn filter_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        Ok(mask)
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask_fut: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        let expr = expr.clone();
        let row_len =
            usize::try_from(row_range.end - row_range.start).vortex_expect("row len fits in usize");

        let is_nullable = self.dtype().is_nullable();
        let validity_idx = if is_nullable { Some(0) } else { None };
        let offsets_idx = if is_nullable { 1 } else { 0 };
        let elements_idx = if is_nullable { 2 } else { 1 };

        let validity_reader = validity_idx
            .map(|idx| self.lazy_children.get(idx))
            .transpose()?
            .cloned();
        let offsets_reader = self.lazy_children.get(offsets_idx)?.clone();
        let elements_reader = self.lazy_children.get(elements_idx)?.clone();

        let offsets_range = row_range.start..(row_range.end + 1);
        let offsets_fut = offsets_reader.projection_evaluation(
            &offsets_range,
            &root(),
            MaskFuture::new_true((row_len + 1) as usize),
        )?;

        let validity_fut = validity_reader
            .map(|r| r.projection_evaluation(row_range, &root(), MaskFuture::new_true(row_len)))
            .transpose()?;

        Ok(Box::pin(async move {
            let offsets_arr = offsets_fut.await?;
            let canonical_offsets = offsets_arr.to_primitive();

            let first_offset_scalar = canonical_offsets.scalar_at(0);
            let first_offset: u64 = first_offset_scalar
                .as_primitive()
                .as_::<u64>()
                .vortex_expect("offset must be u64");
            let last_offset: u64 = canonical_offsets
                .scalar_at(canonical_offsets.len() - 1)
                .as_primitive()
                .as_::<u64>()
                .vortex_expect("offset must be u64");

            let elements_range = first_offset..last_offset;
            let elements_fut = elements_reader.projection_evaluation(
                &elements_range,
                &root(),
                MaskFuture::new_true((elements_range.end - elements_range.start) as usize),
            )?;

            let elements_arr = elements_fut.await?;
            let rebased_offsets = sub_scalar(&offsets_arr, first_offset_scalar)?;

            let validity = if let Some(vf) = validity_fut {
                let v_arr = vf.await?;
                Validity::Array(v_arr)
            } else {
                Validity::NonNullable
            };

            let list_arr = ListArray::try_new(elements_arr, rebased_offsets, validity)?;

            let mask = mask_fut.await?;
            let mut array = list_arr.into_array();

            if !mask.all_true() {
                array = filter(array.as_ref(), &mask)?;
            }

            Ok(if *USE_VORTEX_OPERATORS {
                array.as_ref().apply(&expr)?
            } else if expr.is::<Root>() {
                array
            } else {
                expr.evaluate(&array)?
            })
        }))
    }
}
