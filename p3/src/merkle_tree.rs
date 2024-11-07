use crate::{
    keccak::Keccak256Hash,
    symmetric::{CompressionFunctionFromHasher, GenericHasher},
};

pub use p3_merkle_tree::*;

pub type Keccak256MerkleTreeMmcs<F, const CHUNK: usize = 2> = MerkleTreeMmcs<
    F,
    u8,
    GenericHasher<Keccak256Hash>,
    CompressionFunctionFromHasher<Keccak256Hash, CHUNK, 32>,
    32,
>;

pub fn keccak256_merkle_tree<F, const CHUNK: usize>() -> Keccak256MerkleTreeMmcs<F, CHUNK> {
    MerkleTreeMmcs::new(
        GenericHasher::new(Keccak256Hash),
        CompressionFunctionFromHasher::new(Keccak256Hash),
    )
}
