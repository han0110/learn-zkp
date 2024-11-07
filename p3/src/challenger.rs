use crate::{
    field::{AbstractExtensionField, ExtensionField, Field, FromUniformBytes},
    keccak::Keccak256Hash,
    symmetric::CryptographicHasher,
};
use core::{fmt::Debug, marker::PhantomData};
use serde::Serialize;

pub use p3_challenger::*;

pub trait FieldChallengerExt<F: Field>: FieldChallenger<F> {
    fn observe_ext_slice<EF: AbstractExtensionField<F>>(&mut self, exts: &[EF]) {
        exts.iter()
            .for_each(|ext| self.observe_slice(ext.as_base_slice()));
    }
}

impl<F: Field, T: FieldChallenger<F>> FieldChallengerExt<F> for T {}

#[derive(Clone, Debug)]
pub struct GenericChallenger<F, H>
where
    H: CryptographicHasher<u8, [u8; 32]>,
{
    inner: HashChallenger<u8, H, 32>,
    _marker: PhantomData<F>,
}

impl<F, H> GenericChallenger<F, H>
where
    H: CryptographicHasher<u8, [u8; 32]>,
{
    pub fn new(initial_state: Vec<u8>, hasher: H) -> Self {
        Self {
            inner: HashChallenger::new(initial_state, hasher),
            _marker: PhantomData,
        }
    }
}

impl<F> GenericChallenger<F, Keccak256Hash> {
    pub fn keccak256() -> Self {
        Self::new(Vec::new(), Keccak256Hash)
    }
}

impl<F, T, H> CanObserve<T> for GenericChallenger<F, H>
where
    T: Serialize,
    H: CryptographicHasher<u8, [u8; 32]>,
{
    fn observe(&mut self, value: T) {
        self.inner
            .observe_slice(&bincode::serialize(&value).unwrap());
    }
}

impl<F, E, H> CanSample<E> for GenericChallenger<F, H>
where
    F: Field + FromUniformBytes,
    E: ExtensionField<F>,
    H: CryptographicHasher<u8, [u8; 32]>,
{
    fn sample(&mut self) -> E {
        let sample_base = |inner: &mut HashChallenger<u8, H, 32>| {
            F::from_uniform_bytes(|bytes| bytes.fill_with(|| inner.sample()))
        };
        E::from_base_fn(|_| sample_base(&mut self.inner))
    }
}

impl<F, H> CanSampleBits<usize> for GenericChallenger<F, H>
where
    H: CryptographicHasher<u8, [u8; 32]>,
{
    fn sample_bits(&mut self, bits: usize) -> usize {
        usize::from_le_bytes(self.inner.sample_array()) & ((1 << bits) - 1)
    }
}

impl<F, H> FieldChallenger<F> for GenericChallenger<F, H>
where
    F: Sync + Field + FromUniformBytes,
    H: Sync + CryptographicHasher<u8, [u8; 32]>,
{
}
