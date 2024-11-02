use crate::field::FieldExt;
use core::{fmt::Debug,marker::PhantomData};
use p3_challenger::{CanObserve, CanSample, CanSampleBits, FieldChallenger, HashChallenger};
use p3_field::ExtensionField;
use p3_symmetric::CryptographicHasher;

#[derive(Debug)]
pub struct FieldExtChallenger<F, H>
where
    H: CryptographicHasher<u8, [u8; 32]>,
{
    inner: HashChallenger<u8, H, 32>,
    _marker: PhantomData<F>,
}

impl<F, H> FieldExtChallenger<F, H>
where
    H: CryptographicHasher<u8, [u8; 32]>,
{
    pub fn new(inner: HashChallenger<u8, H, 32>) -> Self {
        Self {
            inner,
            _marker: PhantomData,
        }
    }
}

impl<F, E, H> CanObserve<E> for FieldExtChallenger<F, H>
where
    F: FieldExt,
    E: ExtensionField<F>,
    H: CryptographicHasher<u8, [u8; 32]>,
{
    fn observe(&mut self, value: E) {
        value.as_base_slice().iter().for_each(|base| {
            self.inner.observe_slice(base.to_repr().as_ref());
        });
    }
}

impl<F, E, H> CanSample<E> for FieldExtChallenger<F, H>
where
    F: FieldExt,
    E: ExtensionField<F>,
    H: CryptographicHasher<u8, [u8; 32]>,
{
    fn sample(&mut self) -> E {
        let sample_base = |inner: &mut HashChallenger<u8, H, 32>| {
            let mut repr = F::Repr::default();
            let len = repr.as_ref().len();
            loop {
                repr.as_mut().copy_from_slice(&inner.sample_vec(len));
                if let Some(value) = F::try_from_repr(repr) {
                    return value;
                }
            }
        };
        E::from_base_fn(|_| sample_base(&mut self.inner))
    }
}

impl<F, H> CanSampleBits<usize> for FieldExtChallenger<F, H>
where
    H: CryptographicHasher<u8, [u8; 32]>,
{
    fn sample_bits(&mut self, bits: usize) -> usize {
        usize::from_le_bytes(self.inner.sample_array()) & ((1 << bits) - 1)
    }
}

impl<F, E, H> FieldChallenger<E> for FieldExtChallenger<F, H>
where
    F: Sync + FieldExt,
    E: Sync + ExtensionField<F>,
    H: Sync + CryptographicHasher<u8, [u8; 32]>,
{
}
