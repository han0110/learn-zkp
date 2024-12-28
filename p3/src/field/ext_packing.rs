use crate::field::{ExtensionField, Field, FieldExtensionAlgebra, PackedValue};
use util::zip;

pub trait ExtPackedValue<F: Field, E: ExtensionField<F, ExtensionPacking = Self>>:
    FieldExtensionAlgebra<F::Packing>
{
    fn ext_sum(&self) -> E {
        self.ext_unpack().into_iter().sum()
    }

    fn ext_broadcast(ext: E) -> Self {
        E::ExtensionPacking::from_base_fn(|i| F::Packing::from(ext.as_base_slice()[i]))
    }

    fn ext_pack(buf: &[E]) -> Self {
        debug_assert_eq!(buf.len(), F::Packing::WIDTH);
        E::ExtensionPacking::from_base_fn(|i| F::Packing::from_fn(|j| buf[j].as_base_slice()[i]))
    }

    fn ext_pack_slice_into(packed: &mut [Self], buf: &[E]) {
        debug_assert_eq!(buf.len() % F::Packing::WIDTH, 0);
        zip!(packed, buf.chunks(F::Packing::WIDTH))
            .for_each(|(packed, buf)| *packed = Self::ext_pack(buf));
    }

    fn ext_pack_slice(buf: &[E]) -> Vec<Self> {
        debug_assert_eq!(buf.len() % F::Packing::WIDTH, 0);
        buf.chunks(F::Packing::WIDTH).map(Self::ext_pack).collect()
    }

    fn ext_unpack_slice_into(buf: &mut [E], packed: &[Self]) {
        debug_assert_eq!(buf.len(), packed.len() * F::Packing::WIDTH);
        zip!(packed, buf.chunks_mut(F::Packing::WIDTH)).for_each(|(packed, buf)| {
            (0..F::Packing::WIDTH)
                .for_each(|i| buf[i] = E::from_base_fn(|j| packed.as_base_slice()[j].as_slice()[i]))
        });
    }

    fn ext_unpack(&self) -> Vec<E> {
        (0..F::Packing::WIDTH)
            .map(|i| E::from_base_fn(|j| self.as_base_slice()[j].as_slice()[i]))
            .collect()
    }

    fn ext_unpack_slice(packed: &[Self]) -> Vec<E> {
        packed
            .iter()
            .flat_map(|packed| {
                (0..F::Packing::WIDTH)
                    .map(|i| E::from_base_fn(|j| packed.as_base_slice()[j].as_slice()[i]))
            })
            .collect()
    }
}

impl<
        F: Field,
        E: ExtensionField<F, ExtensionPacking = T>,
        T: FieldExtensionAlgebra<F::Packing>,
    > ExtPackedValue<F, E> for T
{
}
