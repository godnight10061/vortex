// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod reader;
#[cfg(test)]
mod test_robustness;
#[cfg(test)]
mod tests;
pub mod writer;

use std::sync::Arc;

use reader::ListReader;
use vortex_array::ArrayContext;
use vortex_array::DeserializeMetadata;
use vortex_array::ProstMetadata;
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::LayoutChildType;
use crate::LayoutEncodingRef;
use crate::LayoutId;
use crate::LayoutReaderRef;
use crate::LayoutRef;
use crate::VTable;
use crate::children::LayoutChildren;
use crate::children::OwnedLayoutChildren;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::vtable;

vtable!(List);

impl VTable for ListVTable {
    type Layout = ListLayout;
    type Encoding = ListLayoutEncoding;
    type Metadata = ProstMetadata<ListLayoutMetadata>;

    fn id(_encoding: &Self::Encoding) -> LayoutId {
        LayoutId::new_ref("vortex.list")
    }

    fn encoding(_layout: &Self::Layout) -> LayoutEncodingRef {
        LayoutEncodingRef::new_ref(ListLayoutEncoding.as_ref())
    }

    fn row_count(layout: &Self::Layout) -> u64 {
        layout.row_count
    }

    fn dtype(layout: &Self::Layout) -> &DType {
        &layout.dtype
    }

    fn metadata(layout: &Self::Layout) -> Self::Metadata {
        ProstMetadata(ListLayoutMetadata {
            offsets_ptype: PType::try_from(layout.offsets_dtype()).vortex_expect("ptype") as i32,
        })
    }

    fn segment_ids(_layout: &Self::Layout) -> Vec<SegmentId> {
        vec![]
    }

    fn nchildren(layout: &Self::Layout) -> usize {
        if layout.dtype.is_nullable() { 3 } else { 2 }
    }

    fn child(layout: &Self::Layout, index: usize) -> VortexResult<LayoutRef> {
        let is_nullable = layout.dtype.is_nullable();
        let child_dtype = match (is_nullable, index) {
            (true, 0) => DType::Bool(Nullability::NonNullable),
            (true, 1) | (false, 0) => layout.offsets_dtype().clone(),
            (true, 2) | (false, 1) => layout
                .dtype
                .as_list_element_opt()
                .vortex_expect("list element")
                .as_ref()
                .clone(),
            _ => vortex_bail!("Child index out of bounds: {}", index),
        };

        layout.children.child(index, &child_dtype)
    }

    fn child_type(layout: &Self::Layout, idx: usize) -> LayoutChildType {
        let is_nullable = layout.dtype.is_nullable();
        match (is_nullable, idx) {
            (true, 0) => LayoutChildType::Auxiliary("validity".into()),
            (true, 1) | (false, 0) => LayoutChildType::Auxiliary("offsets".into()),
            (true, 2) | (false, 1) => LayoutChildType::Auxiliary("elements".into()),
            _ => LayoutChildType::Auxiliary("unknown".into()),
        }
    }

    fn new_reader(
        layout: &Self::Layout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(ListReader::try_new(
            layout.clone(),
            name,
            segment_source,
            session.clone(),
        )?))
    }

    fn build(
        _encoding: &Self::Encoding,
        dtype: &DType,
        row_count: u64,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        _segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
        _ctx: ArrayContext,
    ) -> VortexResult<Self::Layout> {
        vortex_ensure!(matches!(dtype, DType::List(..)), "Expected list dtype");

        let expected_children = 2 + (dtype.is_nullable() as usize);
        vortex_ensure!(
            children.nchildren() == expected_children,
            "List layout has {} children, but expected {}",
            children.nchildren(),
            expected_children
        );

        let offsets_ptype =
            PType::try_from(metadata.offsets_ptype).map_err(|_| vortex_err!("Invalid PType"))?;
        let offsets_dtype = DType::Primitive(offsets_ptype, Nullability::NonNullable);

        Ok(ListLayout {
            row_count,
            dtype: dtype.clone(),
            offsets_dtype,
            children: children.to_arc(),
        })
    }

    fn with_children(layout: &mut Self::Layout, children: Vec<LayoutRef>) -> VortexResult<()> {
        let expected_children = 2 + (layout.dtype.is_nullable() as usize);
        vortex_ensure!(
            children.len() == expected_children,
            "ListLayout expects {} children, got {}",
            expected_children,
            children.len()
        );

        layout.children = OwnedLayoutChildren::layout_children(children);
        Ok(())
    }
}

#[derive(Debug)]
pub struct ListLayoutEncoding;

#[derive(Clone, Debug)]
pub struct ListLayout {
    row_count: u64,
    dtype: DType,
    offsets_dtype: DType,
    children: Arc<dyn LayoutChildren>,
}

impl ListLayout {
    pub fn new(
        row_count: u64,
        dtype: DType,
        offsets_dtype: DType,
        children: Vec<LayoutRef>,
    ) -> Self {
        Self {
            row_count,
            dtype,
            offsets_dtype,
            children: OwnedLayoutChildren::layout_children(children),
        }
    }

    #[inline]
    pub fn row_count(&self) -> u64 {
        self.row_count
    }

    #[inline]
    pub fn children(&self) -> &Arc<dyn LayoutChildren> {
        &self.children
    }

    #[inline]
    pub fn offsets_dtype(&self) -> &DType {
        &self.offsets_dtype
    }

    pub fn validity(&self) -> VortexResult<Option<LayoutRef>> {
        if self.dtype.is_nullable() {
            Ok(Some(
                self.children
                    .child(0, &DType::Bool(Nullability::NonNullable))?,
            ))
        } else {
            Ok(None)
        }
    }

    pub fn offsets(&self) -> VortexResult<LayoutRef> {
        let idx = if self.dtype.is_nullable() { 1 } else { 0 };
        self.children.child(idx, &self.offsets_dtype)
    }

    pub fn elements(&self) -> VortexResult<LayoutRef> {
        let idx = if self.dtype.is_nullable() { 2 } else { 1 };
        self.children.child(
            idx,
            self.dtype
                .as_list_element_opt()
                .vortex_expect("list element")
                .as_ref(),
        )
    }
}

#[derive(prost::Message)]
pub struct ListLayoutMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    pub offsets_ptype: i32,
}
