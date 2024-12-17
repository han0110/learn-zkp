use core::fmt::Debug;
use p3::{challenger::CanObserve, field::Field};

pub mod basefold;

pub type MultiPolyEvalPoint<F> = Vec<F>;

pub type UniPolyEvalPoint<F> = F;

#[derive(Clone, Debug)]
pub struct PolyEvals<E, P> {
    pub values: Vec<E>,
    pub point: P,
}

pub trait PolyCommitScheme<E: Field, C: CanObserve<Self::Commitment>>: Sized + Debug {
    type Config: Debug;
    type Data: Debug;
    type Commitment: Clone + Debug;
    type CommitmentData: Debug + AsRef<Self::Commitment>;
    type Point: Debug;
    type Proof: Debug;
    type Error: Debug;

    fn setup(config: Self::Config) -> Result<Self, Self::Error>;

    fn commit(&self, data: Self::Data) -> Result<Self::CommitmentData, Self::Error>;

    fn data<'a>(&self, comm_data: &'a Self::CommitmentData) -> &'a Self::Data;

    fn eval(&self, data: &Self::Data, point: &Self::Point) -> Vec<E>;

    fn open(
        &self,
        comm_data: Self::CommitmentData,
        evals: &[PolyEvals<E, Self::Point>],
        challenger: C,
    ) -> Result<Self::Proof, Self::Error>;

    fn verify(
        &self,
        comm: Self::Commitment,
        evals: &[PolyEvals<E, Self::Point>],
        proof: &Self::Proof,
        challenger: C,
    ) -> Result<(), Self::Error>;
}

#[cfg(test)]
pub mod test {
    use crate::{MultiPolyEvalPoint, PolyCommitScheme, PolyEvals};
    use p3::{
        challenger::{CanObserve, FieldChallengerExt, GenericChallenger},
        field::{ExtensionField, Field, FromUniformBytes},
        keccak::Keccak256Hash,
    };
    use rand::{rngs::StdRng, SeedableRng};
    use util::Itertools;

    pub fn run_multi_poly_commit_scheme<
        F: Field + FromUniformBytes,
        E: ExtensionField<F> + FromUniformBytes,
        P: PolyCommitScheme<E, GenericChallenger<F, Keccak256Hash>, Point = MultiPolyEvalPoint<E>>,
    >(
        f: impl Fn(usize, usize, &mut StdRng) -> (P::Config, P::Data),
    ) where
        GenericChallenger<F, Keccak256Hash>: CanObserve<P::Commitment>,
    {
        let mut rng = StdRng::from_entropy();
        for ((num_vars, num_polys), num_points) in
            // TODO: Support multi-point
            // (0..10).cartesian_product(1..10).cartesian_product(1..10)
            (0..10).cartesian_product(1..10).cartesian_product(1..2)
        {
            let (config, polys) = f(num_vars, num_polys, &mut rng);

            let pcs = P::setup(config).unwrap();
            let mut challenger = GenericChallenger::keccak256();

            let comm_data = pcs.commit(polys).unwrap();
            let comm = comm_data.as_ref().clone();

            challenger.observe(comm.clone());

            let evals = (0..num_points)
                .map(|_| E::random_vec(num_vars, &mut rng))
                .map(|point| {
                    let values = pcs.eval(pcs.data(&comm_data), &point);
                    PolyEvals { point, values }
                })
                .collect_vec();

            evals
                .iter()
                .for_each(|evals| challenger.observe_ext_slice(&evals.values));

            let proof = pcs.open(comm_data, &evals, challenger.clone()).unwrap();

            pcs.verify(comm, &evals, &proof, challenger.clone())
                .unwrap();
        }
    }
}
