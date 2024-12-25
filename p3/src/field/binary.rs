use crate::field::Field;

pub trait BinaryField: Field {
    fn basis(i: usize) -> Self;
}
