use crate::izip;
use core::{array::from_fn, iter::repeat_with};
use p3_baby_bear::BabyBear;
use p3_field::{
    extension::Complex, AbstractExtensionField, AbstractField, ExtensionField, Field, PackedValue,
    PrimeField32, PrimeField64,
};
use p3_goldilocks::Goldilocks;
use p3_mersenne_31::Mersenne31;
use rand::RngCore;

pub trait ExtPackedValue<F: Field, E: ExtensionField<F, ExtensionPacking = Self>>:
    AbstractExtensionField<F::Packing>
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
        izip!(packed, buf.chunks(F::Packing::WIDTH))
            .for_each(|(packed, buf)| *packed = Self::ext_pack(buf));
    }

    fn ext_pack_slice(buf: &[E]) -> Vec<Self> {
        debug_assert_eq!(buf.len() % F::Packing::WIDTH, 0);
        buf.chunks(F::Packing::WIDTH).map(Self::ext_pack).collect()
    }

    fn ext_unpack_slice_into(buf: &mut [E], packed: &[Self]) {
        debug_assert_eq!(buf.len(), packed.len() * F::Packing::WIDTH);
        izip!(packed, buf.chunks_mut(F::Packing::WIDTH)).for_each(|(packed, buf)| {
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
        T: AbstractExtensionField<F::Packing>,
    > ExtPackedValue<F, E> for T
{
}

pub trait FieldSlice<F: Field>: AsRef<[F]> + AsMut<[F]> {
    fn slice_add_assign(&mut self, rhs: &[F]) {
        debug_assert_eq!(self.as_ref().len(), rhs.len());
        let (packed_lhs, suffix_lhs) = F::Packing::pack_slice_with_suffix_mut(self.as_mut());
        let (packed_rhs, suffix_rhs) = F::Packing::pack_slice_with_suffix(rhs);
        izip!(packed_lhs, packed_rhs).for_each(|(lhs, rhs)| *lhs += *rhs);
        izip!(suffix_lhs, suffix_rhs).for_each(|(lhs, rhs)| *lhs += *rhs);
    }

    fn slice_sub_assign(&mut self, rhs: &[F]) {
        debug_assert_eq!(self.as_ref().len(), rhs.len());
        let (packed_lhs, suffix_lhs) = F::Packing::pack_slice_with_suffix_mut(self.as_mut());
        let (packed_rhs, suffix_rhs) = F::Packing::pack_slice_with_suffix(rhs);
        izip!(packed_lhs, packed_rhs).for_each(|(lhs, rhs)| *lhs -= *rhs);
        izip!(suffix_lhs, suffix_rhs).for_each(|(lhs, rhs)| *lhs -= *rhs);
    }

    fn slice_sub(&mut self, lhs: &[F], rhs: &[F]) {
        debug_assert_eq!(self.as_ref().len(), rhs.len());
        debug_assert_eq!(self.as_ref().len(), lhs.len());
        let (packed_out, suffix_out) = F::Packing::pack_slice_with_suffix_mut(self.as_mut());
        let (packed_lhs, suffix_lhs) = F::Packing::pack_slice_with_suffix(lhs);
        let (packed_rhs, suffix_rhs) = F::Packing::pack_slice_with_suffix(rhs);
        izip!(packed_out, packed_lhs, packed_rhs).for_each(|(out, lhs, rhs)| *out = *lhs - *rhs);
        izip!(suffix_out, suffix_lhs, suffix_rhs).for_each(|(out, lhs, rhs)| *out = *lhs - *rhs);
    }

    fn slice_add_scaled_assign(&mut self, rhs: &[F], scalar: F) {
        debug_assert_eq!(self.as_ref().len(), rhs.len());
        let (packed_lhs, suffix_lhs) = F::Packing::pack_slice_with_suffix_mut(self.as_mut());
        let (packed_rhs, suffix_rhs) = F::Packing::pack_slice_with_suffix(rhs);
        izip!(packed_lhs, packed_rhs).for_each(|(lhs, rhs)| *lhs += *rhs * scalar);
        izip!(suffix_lhs, suffix_rhs).for_each(|(lhs, rhs)| *lhs += *rhs * scalar);
    }

    fn slice_mul(&mut self, lhs: &[F], rhs: &[F]) {
        debug_assert_eq!(self.as_ref().len(), rhs.len());
        debug_assert_eq!(self.as_ref().len(), lhs.len());
        let (packed_out, suffix_out) = F::Packing::pack_slice_with_suffix_mut(self.as_mut());
        let (packed_lhs, suffix_lhs) = F::Packing::pack_slice_with_suffix(lhs);
        let (packed_rhs, suffix_rhs) = F::Packing::pack_slice_with_suffix(rhs);
        izip!(packed_out, packed_lhs, packed_rhs).for_each(|(out, lhs, rhs)| *out = *lhs * *rhs);
        izip!(suffix_out, suffix_lhs, suffix_rhs).for_each(|(out, lhs, rhs)| *out = *lhs * *rhs);
    }

    fn slice_dit_butterfly(&self, lhs: &mut [F], rhs: &mut [F]) {
        debug_assert_eq!(self.as_ref().len(), lhs.len());
        debug_assert_eq!(self.as_ref().len(), rhs.len());
        let (packed_twd, suffix_twd) = F::Packing::pack_slice_with_suffix(self.as_ref());
        let (packed_lhs, suffix_lhs) = F::Packing::pack_slice_with_suffix_mut(lhs);
        let (packed_rhs, suffix_rhs) = F::Packing::pack_slice_with_suffix_mut(rhs);
        izip!(packed_twd, packed_lhs, packed_rhs).for_each(dit_butterfly);
        izip!(suffix_twd, suffix_lhs, suffix_rhs).for_each(dit_butterfly);

        #[inline]
        fn dit_butterfly<F: AbstractField + Copy>((t, a, b): (&F, &mut F, &mut F)) {
            let t_b = *t * *b;
            (*a, *b) = (*a + t_b, *a - t_b);
        }
    }
}

impl<F: Field, T: AsRef<[F]> + AsMut<[F]>> FieldSlice<F> for T {}

pub trait FromUniformBytes: Sized {
    type Bytes: Copy + Default + AsRef<[u8]> + AsMut<[u8]>;

    fn from_uniform_bytes(mut fill: impl FnMut(&mut [u8])) -> Self {
        let mut bytes = Self::Bytes::default();
        loop {
            fill(bytes.as_mut());
            if let Some(value) = Self::try_from_uniform_bytes(bytes) {
                return value;
            }
        }
    }

    fn try_from_uniform_bytes(bytes: Self::Bytes) -> Option<Self>;

    fn random(mut rng: impl RngCore) -> Self {
        Self::from_uniform_bytes(|bytes| rng.fill_bytes(bytes.as_mut()))
    }

    fn random_vec(n: usize, mut rng: impl RngCore) -> Vec<Self> {
        repeat_with(|| Self::random(&mut rng)).take(n).collect()
    }
}

impl FromUniformBytes for Mersenne31 {
    type Bytes = [u8; 4];

    fn try_from_uniform_bytes(bytes: [u8; 4]) -> Option<Self> {
        let value = u32::from_le_bytes(bytes) >> 1;
        let is_canonical = value != Self::ORDER_U32;
        is_canonical.then(|| Self::from_canonical_u32(value))
    }
}

impl FromUniformBytes for BabyBear {
    type Bytes = [u8; 4];

    fn try_from_uniform_bytes(bytes: [u8; 4]) -> Option<Self> {
        let value = u32::from_le_bytes(bytes) >> 1;
        let is_canonical = value < Self::ORDER_U32;
        is_canonical.then(|| Self::from_canonical_u32(value))
    }
}

impl FromUniformBytes for Goldilocks {
    type Bytes = [u8; 8];

    fn try_from_uniform_bytes(bytes: [u8; 8]) -> Option<Self> {
        let value = u64::from_le_bytes(bytes);
        let is_canonical = value < Self::ORDER_U64;
        is_canonical.then(|| Self::from_canonical_u64(value))
    }
}

macro_rules! impl_from_uniform_bytes_for_binomial_extension {
    ($base:ty, $degree:literal) => {
        impl FromUniformBytes for p3_field::extension::BinomialExtensionField<$base, $degree> {
            type Bytes = [u8; <$base as FromUniformBytes>::Bytes::WIDTH * $degree];

            fn try_from_uniform_bytes(bytes: Self::Bytes) -> Option<Self> {
                Some(p3_field::AbstractExtensionField::from_base_slice(
                    &array_try_from_uniform_bytes::<
                        $base,
                        { <$base as FromUniformBytes>::Bytes::WIDTH },
                        $degree,
                    >(&bytes)?,
                ))
            }
        }
    };
}

impl_from_uniform_bytes_for_binomial_extension!(Mersenne31, 3);
impl_from_uniform_bytes_for_binomial_extension!(Complex<Mersenne31>, 2);
impl_from_uniform_bytes_for_binomial_extension!(Complex<Mersenne31>, 3);
impl_from_uniform_bytes_for_binomial_extension!(BabyBear, 4);
impl_from_uniform_bytes_for_binomial_extension!(BabyBear, 5);
impl_from_uniform_bytes_for_binomial_extension!(Goldilocks, 2);

impl FromUniformBytes for Complex<Mersenne31> {
    type Bytes = [u8; 8];

    fn try_from_uniform_bytes(bytes: Self::Bytes) -> Option<Self> {
        let [real, imag] = array_try_from_uniform_bytes(&bytes)?;
        Some(Complex::new(real, imag))
    }
}

fn array_try_from_uniform_bytes<
    F: Copy + Default + FromUniformBytes<Bytes = [u8; W]>,
    const W: usize,
    const N: usize,
>(
    bytes: &[u8],
) -> Option<[F; N]> {
    let mut array = [F::default(); N];
    for i in 0..N {
        array[i] = F::try_from_uniform_bytes(from_fn(|j| bytes[i * W + j]))?;
    }
    Some(array)
}
