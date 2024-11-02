use core::iter::repeat_with;
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, Field, PrimeField32, PrimeField64};
use p3_goldilocks::Goldilocks;
use p3_mersenne_31::Mersenne31;
use rand::{distributions::Standard, Rng, RngCore};

pub trait FieldExt: Field {
    type Repr: Copy + Default + AsRef<[u8]> + AsMut<[u8]>;

    fn try_from_repr(bytes: Self::Repr) -> Option<Self>;

    fn to_repr(&self) -> Self::Repr;

    fn random(rng: impl RngCore) -> Self;

    fn random_vec(n: usize, mut rng: impl RngCore) -> Vec<Self> {
        repeat_with(|| Self::random(&mut rng)).take(n).collect()
    }
}

impl FieldExt for Mersenne31 {
    type Repr = [u8; 4];

    fn try_from_repr(bytes: [u8; 4]) -> Option<Self> {
        let value = u32::from_le_bytes(bytes) >> 1;
        let is_canonical = value != Self::ORDER_U32;
        is_canonical.then(|| Self::from_canonical_u32(value))
    }

    fn to_repr(&self) -> Self::Repr {
        self.as_canonical_u32().to_le_bytes()
    }

    fn random(mut rng: impl RngCore) -> Self {
        rng.sample(Standard)
    }
}

impl FieldExt for BabyBear {
    type Repr = [u8; 4];

    fn try_from_repr(bytes: [u8; 4]) -> Option<Self> {
        let value = u32::from_le_bytes(bytes) >> 1;
        let is_canonical = value < Self::ORDER_U32;
        is_canonical.then(|| Self::from_canonical_u32(value))
    }

    fn to_repr(&self) -> Self::Repr {
        self.as_canonical_u32().to_le_bytes()
    }

    fn random(mut rng: impl RngCore) -> Self {
        rng.sample(Standard)
    }
}

impl FieldExt for Goldilocks {
    type Repr = [u8; 8];

    fn try_from_repr(bytes: [u8; 8]) -> Option<Self> {
        let value = u64::from_le_bytes(bytes);
        let is_canonical = value < Self::ORDER_U64;
        is_canonical.then(|| Self::from_canonical_u64(value))
    }

    fn to_repr(&self) -> Self::Repr {
        self.as_canonical_u64().to_le_bytes()
    }

    fn random(mut rng: impl RngCore) -> Self {
        rng.sample(Standard)
    }
}
