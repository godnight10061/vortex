// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::ArrayContext;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::ToCanonical;
use vortex_array::arrays::{ListArray, ListVTable, StructArray};
use vortex_array::expr::root;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_io::runtime::single::block_on;
use vortex_mask::Mask;

use crate::LayoutStrategy;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::layouts::list::ListVTable as ListLayoutVTable;
use crate::layouts::list::writer::ListLayoutStrategy;
use crate::layouts::table::TableStrategy;
use crate::segments::TestSegments;
use crate::sequence::SequenceId;
use crate::sequence::SequentialArrayStreamExt;
use crate::test::SESSION;

#[test]
fn list_roundtrip() {
    block_on(|handle| async {
        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();

        // [[1, 2], [3, 4, 5], []]
        let elements = buffer![1i32, 2, 3, 4, 5].into_array();
        let offsets = buffer![0i32, 2, 5, 5].into_array();
        let array = ListArray::new(elements, offsets, Validity::NonNullable).into_array();

        let strategy = ListLayoutStrategy::new(
            Arc::new(FlatLayoutStrategy::default()),
            Arc::new(FlatLayoutStrategy::default()),
            Arc::new(FlatLayoutStrategy::default()),
        );

        let layout = strategy
            .write_stream(
                ctx,
                segments.clone(),
                array.to_array_stream().sequenced(ptr),
                eof,
                handle,
            )
            .await
            .unwrap();

        assert_eq!(layout.encoding_id().as_ref(), "vortex.list");
        assert_eq!(layout.row_count(), 3);

        let reader = layout.new_reader("".into(), segments, &SESSION).unwrap();
        let result = reader
            .projection_evaluation(&(0..3), &root(), MaskFuture::new_true(3))
            .unwrap()
            .await
            .unwrap();

        assert_eq!(result.len(), 3);
        let list_res = result.as_opt::<ListVTable>().unwrap();
        assert_eq!(
            list_res.elements().to_primitive().as_slice::<i32>(),
            &[1, 2, 3, 4, 5]
        );
        assert_eq!(
            list_res.offsets().to_primitive().as_slice::<u64>(),
            &[0, 2, 5, 5]
        );
    })
}

#[test]
fn list_nullable_roundtrip() {
    block_on(|handle| async {
        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();

        // [[1, 2], NULL, []]
        let elements = buffer![1i32, 2].into_array();
        let offsets = buffer![0i32, 2, 2, 2].into_array();
        let validity = Validity::from_iter([true, false, true]);
        let array = ListArray::new(elements, offsets, validity).into_array();

        let strategy = ListLayoutStrategy::new(
            Arc::new(FlatLayoutStrategy::default()),
            Arc::new(FlatLayoutStrategy::default()),
            Arc::new(FlatLayoutStrategy::default()),
        );

        let layout = strategy
            .write_stream(
                ctx,
                segments.clone(),
                array.to_array_stream().sequenced(ptr),
                eof,
                handle,
            )
            .await
            .unwrap();

        assert_eq!(layout.encoding_id().as_ref(), "vortex.list");
        assert_eq!(layout.row_count(), 3);

        let reader = layout.new_reader("".into(), segments, &SESSION).unwrap();
        let result = reader
            .projection_evaluation(&(0..3), &root(), MaskFuture::new_true(3))
            .unwrap()
            .await
            .unwrap();

        assert_eq!(result.len(), 3);
        assert!(result.dtype().is_nullable());
        let list_res = result.as_opt::<ListVTable>().unwrap();
        assert_eq!(
            list_res.elements().to_primitive().as_slice::<i32>(),
            &[1, 2]
        );
        assert_eq!(
            list_res.offsets().to_primitive().as_slice::<u64>(),
            &[0, 2, 2, 2]
        );
        assert!(
            result
                .validity_mask()
                .to_bit_buffer()
                .iter()
                .collect::<Vec<_>>()[0]
        );
        assert!(
            !result
                .validity_mask()
                .to_bit_buffer()
                .iter()
                .collect::<Vec<_>>()[1]
        );
        assert!(
            result
                .validity_mask()
                .to_bit_buffer()
                .iter()
                .collect::<Vec<_>>()[2]
        );
    })
}

#[test]
fn list_struct_roundtrip() {
    block_on(|handle| async {
        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();

        // [[{"a": 1}, {"a": 2}], [{"a": 3}]]
        let elements = StructArray::from_fields(&[("a", buffer![1i32, 2, 3].into_array())])
            .unwrap()
            .into_array();
        let offsets = buffer![0i32, 2, 3].into_array();
        let array = ListArray::new(elements, offsets, Validity::NonNullable).into_array();

        // Use TableStrategy which should now use ListLayoutStrategy for the list column
        let strategy = TableStrategy::default();

        let layout = strategy
            .write_stream(
                ctx,
                segments.clone(),
                array.to_array_stream().sequenced(ptr),
                eof,
                handle,
            )
            .await
            .unwrap();

        // It should be a ListLayout
        assert_eq!(layout.encoding_id().as_ref(), "vortex.list");

        // Its elements child should be a StructLayout
        let _list_layout = layout.as_opt::<ListLayoutVTable>().unwrap();
        // Index of elements child is 1 (non-nullable list)
        let elements_layout = layout.child(1).unwrap();
        assert_eq!(elements_layout.encoding_id().as_ref(), "vortex.struct");

        let reader = layout.new_reader("".into(), segments, &SESSION).unwrap();
        let result = reader
            .projection_evaluation(&(0..2), &root(), MaskFuture::new_true(2))
            .unwrap()
            .await
            .unwrap();

        assert_eq!(result.len(), 2);
        let list_res = result.as_opt::<ListVTable>().unwrap();
        let res_elements = list_res.elements().to_struct();
        assert_eq!(
            res_elements
                .field_by_name("a")
                .unwrap()
                .to_primitive()
                .as_slice::<i32>(),
            &[1, 2, 3]
        );
    })
}

#[test]
fn list_projection_respects_row_mask() {
    block_on(|handle| async {
        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();

        // [[1, 2], [3, 4, 5], []]
        let elements = buffer![1i32, 2, 3, 4, 5].into_array();
        let offsets = buffer![0i32, 2, 5, 5].into_array();
        let array = ListArray::new(elements, offsets, Validity::NonNullable).into_array();

        let strategy = ListLayoutStrategy::new(
            Arc::new(FlatLayoutStrategy::default()),
            Arc::new(FlatLayoutStrategy::default()),
            Arc::new(FlatLayoutStrategy::default()),
        );

        let layout = strategy
            .write_stream(
                ctx,
                segments.clone(),
                array.to_array_stream().sequenced(ptr),
                eof,
                handle,
            )
            .await
            .unwrap();

        let reader = layout.new_reader("".into(), segments, &SESSION).unwrap();
        let mask = MaskFuture::ready(Mask::from_iter([true, false, true]));
        let result = reader
            .projection_evaluation(&(0..3), &root(), mask)
            .unwrap()
            .await
            .unwrap();

        assert_eq!(result.len(), 2);
        let list_res = result.as_opt::<ListVTable>().unwrap();
        assert_eq!(
            list_res.elements().to_primitive().as_slice::<i32>(),
            &[1, 2]
        );
        assert_eq!(
            list_res.offsets().to_primitive().as_slice::<u64>(),
            &[0, 2, 2]
        );
    })
}
