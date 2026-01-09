// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::future::try_join_all;
use futures::pin_mut;
use vortex_array::Array;
use vortex_array::ArrayContext;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::list_from_list_view;
use vortex_array::compute::add_scalar;
use vortex_array::compute::cast;
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_io::kanal_ext::KanalExt;
use vortex_io::runtime::Handle;

use crate::IntoLayout as _;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::layouts::list::ListLayout;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

#[derive(Clone)]
pub struct ListLayoutStrategy {
    offsets: Arc<dyn LayoutStrategy>,
    elements: Arc<dyn LayoutStrategy>,
    validity: Arc<dyn LayoutStrategy>,
}

impl ListLayoutStrategy {
    pub fn new(
        offsets: Arc<dyn LayoutStrategy>,
        elements: Arc<dyn LayoutStrategy>,
        validity: Arc<dyn LayoutStrategy>,
    ) -> Self {
        Self {
            offsets,
            elements,
            validity,
        }
    }
}

#[async_trait]
impl LayoutStrategy for ListLayoutStrategy {
    async fn write_stream(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        mut eof: SequencePointer,
        handle: Handle,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();
        let is_nullable = dtype.is_nullable();
        let element_dtype = dtype
            .as_list_element_opt()
            .vortex_expect("list element")
            .as_ref()
            .clone();
        let offsets_dtype = DType::Primitive(PType::U64, Nullability::NonNullable);
        let offsets_dtype_for_chunks = offsets_dtype.clone();

        let mut current_element_offset: u64 = 0;

        let transposed_stream = stream.map(move |chunk| {
            let (sequence_id, chunk) = chunk?;
            let mut sequence_pointer = sequence_id.descend();
            let list_chunk = list_from_list_view(chunk.to_listview());

            let validity = if is_nullable {
                Some((
                    sequence_pointer.advance(),
                    chunk.validity_mask().into_array(),
                ))
            } else {
                None
            };

            let offsets = cast(list_chunk.offsets().as_ref(), &offsets_dtype_for_chunks)?;
            let offsets_to_send = if current_element_offset == 0 {
                offsets.clone()
            } else {
                let offset_scalar = vortex_scalar::Scalar::primitive(
                    current_element_offset,
                    Nullability::NonNullable,
                );
                let adjusted = add_scalar(&offsets, offset_scalar)?;
                adjusted.slice(1..adjusted.len())
            };

            let offsets_id = sequence_pointer.advance();

            let elements = list_chunk.elements().clone();
            let elements_id = sequence_pointer.advance();

            current_element_offset += elements.len() as u64;

            Ok((
                validity,
                (offsets_id, offsets_to_send),
                (elements_id, elements),
            ))
        });

        let (validity_tx, validity_rx) = if is_nullable {
            let (tx, rx) = kanal::bounded_async(1);
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };
        let (offsets_tx, offsets_rx) = kanal::bounded_async(1);
        let (elements_tx, elements_rx) = kanal::bounded_async(1);

        handle
            .spawn(async move {
                pin_mut!(transposed_stream);
                while let Some(result) = transposed_stream.next().await {
                    match result {
                        Ok((v, o, e)) => {
                            if let (Some(tx), Some(v)) = (&validity_tx, v) {
                                let _ = tx.send(Ok(v)).await;
                            }
                            let _ = offsets_tx.send(Ok(o)).await;
                            let _ = elements_tx.send(Ok(e)).await;
                        }
                        Err(err) => {
                            let err: Arc<VortexError> = Arc::new(err);
                            if let Some(tx) = &validity_tx {
                                let _ = tx.send(Err(VortexError::from(err.clone()))).await;
                            }
                            let _ = offsets_tx.send(Err(VortexError::from(err.clone()))).await;
                            let _ = elements_tx.send(Err(VortexError::from(err.clone()))).await;
                            break;
                        }
                    }
                }
            })
            .detach();

        let mut child_futures = Vec::new();

        if is_nullable {
            let validity_stream = SequentialStreamAdapter::new(
                DType::Bool(Nullability::NonNullable),
                validity_rx.unwrap().into_stream().boxed(),
            )
            .sendable();
            let child_eof = eof.split_off();
            let strategy = self.validity.clone();
            let ctx = ctx.clone();
            let segment_sink = segment_sink.clone();
            child_futures.push(handle.spawn_nested(|h| async move {
                strategy
                    .write_stream(ctx, segment_sink, validity_stream, child_eof, h)
                    .await
            }));
        }

        // Offsets
        {
            let offsets_sequential = SequentialStreamAdapter::new(
                offsets_dtype.clone(),
                offsets_rx.into_stream().boxed(),
            )
            .sendable();

            let child_eof = eof.split_off();
            let strategy = self.offsets.clone();
            let ctx = ctx.clone();
            let segment_sink = segment_sink.clone();
            child_futures.push(handle.spawn_nested(|h| async move {
                strategy
                    .write_stream(ctx, segment_sink, offsets_sequential, child_eof, h)
                    .await
            }));
        }

        // Elements
        {
            let elements_sequential =
                SequentialStreamAdapter::new(element_dtype, elements_rx.into_stream().boxed())
                    .sendable();

            let child_eof = eof.split_off();
            let strategy = self.elements.clone();
            let ctx = ctx.clone();
            let segment_sink = segment_sink.clone();
            child_futures.push(handle.spawn_nested(|h| async move {
                strategy
                    .write_stream(ctx, segment_sink, elements_sequential, child_eof, h)
                    .await
            }));
        }

        let child_layouts = try_join_all(child_futures).await?;

        let row_count = if is_nullable {
            child_layouts[0].row_count()
        } else {
            child_layouts[0].row_count() - 1
        };

        Ok(ListLayout::new(row_count, dtype, offsets_dtype, child_layouts).into_layout())
    }

    fn buffered_bytes(&self) -> u64 {
        self.offsets.buffered_bytes()
            + self.elements.buffered_bytes()
            + self.validity.buffered_bytes()
    }
}
