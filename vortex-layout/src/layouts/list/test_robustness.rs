// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::stream;
use vortex_array::ArrayContext;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::ToCanonical;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListVTable;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::expr::root;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_dtype::Nullability;
use vortex_io::runtime::single::block_on;

use crate::LayoutStrategy;
use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::layouts::list::writer::ListLayoutStrategy;
use crate::segments::TestSegments;
use crate::sequence::SequenceId;
use crate::sequence::SequentialArrayStreamExt;
use crate::test::SESSION;

#[test]
fn test_sliced_list_chunks() {
    block_on(|handle| async {
        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();

        // Create a large list array
        // [[0, 1], [2, 3], [4, 5], [6, 7]]
        let elements = buffer![0i32, 1, 2, 3, 4, 5, 6, 7].into_array();
        let offsets = buffer![0u32, 2, 4, 6, 8].into_array();
        let array = ListArray::new(elements, offsets, Validity::NonNullable).into_array();

        // Slice it into two chunks
        // Chunk 1: [[0, 1], [2, 3]] (Index 0..2)
        let chunk1 = array.slice(0..2);
        // Chunk 2: [[4, 5], [6, 7]] (Index 2..4)
        let chunk2 = array.slice(2..4);

        let stream = stream::iter(vec![Ok(chunk1), Ok(chunk2)]);
        let array_stream = ArrayStreamAdapter::new(array.dtype().clone(), stream);

        let chunked_flat = Arc::new(ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()));
        let strategy =
            ListLayoutStrategy::new(chunked_flat.clone(), chunked_flat.clone(), chunked_flat);

        let layout = strategy
            .write_stream(
                ctx,
                segments.clone(),
                array_stream.sequenced(ptr),
                eof,
                handle,
            )
            .await
            .unwrap();

        assert_eq!(layout.row_count(), 4);

        let reader = layout.new_reader("".into(), segments, &SESSION).unwrap();
        let result = reader
            .projection_evaluation(&(0..4), &root(), MaskFuture::new_true(4))
            .unwrap()
            .await
            .unwrap();

        assert_eq!(result.len(), 4);
        let list_res = result.as_opt::<ListVTable>().unwrap();
        assert_eq!(
            list_res.elements().to_primitive().as_slice::<i32>(),
            &[0, 1, 2, 3, 4, 5, 6, 7]
        );
        assert_eq!(
            list_res.offsets().to_primitive().as_slice::<u64>(),
            &[0, 2, 4, 6, 8]
        );
    })
}

#[test]
fn test_empty_chunks() {
    block_on(|handle| async {
        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();

        // Chunk 1: [[1]]
        let c1_elements = buffer![1i32].into_array();
        let c1_offsets = buffer![0u32, 1].into_array();
        let chunk1 = ListArray::new(c1_elements, c1_offsets, Validity::NonNullable).into_array();

        // Chunk 2: Empty (0 rows)
        let c2_elements = PrimitiveArray::empty::<i32>(Nullability::NonNullable).into_array();
        let c2_offsets = buffer![0u32].into_array();
        let chunk2 = ListArray::new(c2_elements, c2_offsets, Validity::NonNullable).into_array();

        // Chunk 3: [[2]]
        let c3_elements = buffer![2i32].into_array();
        let c3_offsets = buffer![0u32, 1].into_array();
        let chunk3 = ListArray::new(c3_elements, c3_offsets, Validity::NonNullable).into_array();

        let dtype = chunk1.dtype().clone();
        let stream = stream::iter(vec![Ok(chunk1), Ok(chunk2), Ok(chunk3)]);
        let array_stream = ArrayStreamAdapter::new(dtype, stream);

        let chunked_flat = Arc::new(ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()));
        let strategy =
            ListLayoutStrategy::new(chunked_flat.clone(), chunked_flat.clone(), chunked_flat);

        let layout = strategy
            .write_stream(
                ctx,
                segments.clone(),
                array_stream.sequenced(ptr),
                eof,
                handle,
            )
            .await
            .unwrap();

        assert_eq!(layout.row_count(), 2);

        let reader = layout.new_reader("".into(), segments, &SESSION).unwrap();
        let result = reader
            .projection_evaluation(&(0..2), &root(), MaskFuture::new_true(2))
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
            &[0, 1, 2]
        );
    })
}

#[test]
fn test_chunk_with_empty_list() {
    block_on(|handle| async {
        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();

        // Chunk 1: [[1]]
        let c1 = ListArray::new(
            buffer![1i32].into_array(),
            buffer![0u32, 1].into_array(),
            Validity::NonNullable,
        )
        .into_array();

        // Chunk 2: [[]] (One row, empty list)
        let c2 = ListArray::new(
            PrimitiveArray::empty::<i32>(Nullability::NonNullable).into_array(),
            buffer![0u32, 0].into_array(),
            Validity::NonNullable,
        )
        .into_array();

        // Chunk 3: [[2]]
        let c3 = ListArray::new(
            buffer![2i32].into_array(),
            buffer![0u32, 1].into_array(),
            Validity::NonNullable,
        )
        .into_array();

        let dtype = c1.dtype().clone();
        let stream = stream::iter(vec![Ok(c1), Ok(c2), Ok(c3)]);
        let array_stream = ArrayStreamAdapter::new(dtype, stream);

        let chunked_flat = Arc::new(ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()));
        let strategy =
            ListLayoutStrategy::new(chunked_flat.clone(), chunked_flat.clone(), chunked_flat);

        let layout = strategy
            .write_stream(
                ctx,
                segments.clone(),
                array_stream.sequenced(ptr),
                eof,
                handle,
            )
            .await
            .unwrap();

        assert_eq!(layout.row_count(), 3);

        let reader = layout.new_reader("".into(), segments, &SESSION).unwrap();
        let result = reader
            .projection_evaluation(&(0..3), &root(), MaskFuture::new_true(3))
            .unwrap()
            .await
            .unwrap();

        let list_res = result.as_opt::<ListVTable>().unwrap();
        // Elements: [1, 2]
        assert_eq!(
            list_res.elements().to_primitive().as_slice::<i32>(),
            &[1, 2]
        );
        // Offsets: [0, 1, 1, 2]
        assert_eq!(
            list_res.offsets().to_primitive().as_slice::<u64>(),
            &[0, 1, 1, 2]
        );
    })
}

#[test]
fn test_first_chunk_has_no_elements() {
    block_on(|handle| async {
        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();

        // Chunk 1: [[], []] (2 rows, 0 elements)
        let c1 = ListArray::new(
            PrimitiveArray::empty::<i32>(Nullability::NonNullable).into_array(),
            buffer![0u32, 0, 0].into_array(),
            Validity::NonNullable,
        )
        .into_array();

        // Chunk 2: [[1, 2]] (1 row, 2 elements)
        let c2 = ListArray::new(
            buffer![1i32, 2].into_array(),
            buffer![0u32, 2].into_array(),
            Validity::NonNullable,
        )
        .into_array();

        let dtype = c1.dtype().clone();
        let stream = stream::iter(vec![Ok(c1), Ok(c2)]);
        let array_stream = ArrayStreamAdapter::new(dtype, stream);

        let chunked_flat = Arc::new(ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()));
        let strategy =
            ListLayoutStrategy::new(chunked_flat.clone(), chunked_flat.clone(), chunked_flat);

        let layout = strategy
            .write_stream(
                ctx,
                segments.clone(),
                array_stream.sequenced(ptr),
                eof,
                handle,
            )
            .await
            .unwrap();

        assert_eq!(layout.row_count(), 3);

        let reader = layout.new_reader("".into(), segments, &SESSION).unwrap();
        let result = reader
            .projection_evaluation(&(0..3), &root(), MaskFuture::new_true(3))
            .unwrap()
            .await
            .unwrap();

        let list_res = result.as_opt::<ListVTable>().unwrap();
        assert_eq!(
            list_res.elements().to_primitive().as_slice::<i32>(),
            &[1, 2]
        );
        assert_eq!(
            list_res.offsets().to_primitive().as_slice::<u64>(),
            &[0, 0, 0, 2]
        );
    })
}
