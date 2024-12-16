use crate::field::{extension::Complex, FieldAlgebra, PackedValue, PrimeField32, PrimeField64};
use core::{array::from_fn, iter::repeat_with};
use rand::RngCore;

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

impl FromUniformBytes for p3_mersenne_31::Mersenne31 {
    type Bytes = [u8; 4];

    fn try_from_uniform_bytes(bytes: [u8; 4]) -> Option<Self> {
        let value = u32::from_le_bytes(bytes) >> 1;
        let is_canonical = value != Self::ORDER_U32;
        is_canonical.then(|| Self::from_canonical_u32(value))
    }
}

impl FromUniformBytes for p3_baby_bear::BabyBear {
    type Bytes = [u8; 4];

    fn try_from_uniform_bytes(bytes: [u8; 4]) -> Option<Self> {
        let value = u32::from_le_bytes(bytes) >> 1;
        let is_canonical = value < Self::ORDER_U32;
        is_canonical.then(|| Self::from_canonical_u32(value))
    }
}

impl FromUniformBytes for p3_goldilocks::Goldilocks {
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
                Some(p3_field::FieldExtensionAlgebra::from_base_slice(
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

impl_from_uniform_bytes_for_binomial_extension!(p3_mersenne_31::Mersenne31, 3);
impl_from_uniform_bytes_for_binomial_extension!(Complex<p3_mersenne_31::Mersenne31>, 2);
impl_from_uniform_bytes_for_binomial_extension!(Complex<p3_mersenne_31::Mersenne31>, 3);
impl_from_uniform_bytes_for_binomial_extension!(p3_baby_bear::BabyBear, 4);
impl_from_uniform_bytes_for_binomial_extension!(p3_baby_bear::BabyBear, 5);
impl_from_uniform_bytes_for_binomial_extension!(p3_goldilocks::Goldilocks, 2);

impl FromUniformBytes for Complex<p3_mersenne_31::Mersenne31> {
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
