use crate::keccak::Keccak256Hash;
use serde::Serialize;

pub use p3_symmetric::*;

#[derive(Copy, Clone, Debug)]
pub struct GenericHasher<H> {
    inner: H,
}

impl<T, H> CryptographicHasher<T, [u8; 32]> for GenericHasher<H>
where
    T: Clone + Serialize,
    H: CryptographicHasher<u8, [u8; 32]>,
{
    fn hash_iter<I>(&self, input: I) -> [u8; 32]
    where
        I: IntoIterator<Item = T>,
    {
        self.inner.hash_iter(
            input
                .into_iter()
                .flat_map(|x| bincode::serialize(&x).unwrap()),
        )
    }
}

impl<H> GenericHasher<H> {
    pub fn new(inner: H) -> Self {
        Self { inner }
    }
}

impl GenericHasher<Keccak256Hash> {
    pub fn keccak256() -> Self {
        Self::new(Keccak256Hash)
    }
}
