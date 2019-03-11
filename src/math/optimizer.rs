//! Guiding traits to implement iterative optimization algorithms.

/// Enum used to indicate if iterations should continue or stop.
/// Must be returned by the stop_criterion function.
pub enum Continue {
    /// Stop iterations.
    Stop,
    /// Continue iterations.
    Forward,
}

/// An `Optimizer<Observations, S, Delta, Model, PreEval, PartialState, E>`
/// is capable of iteratively minimizing an energy function,
/// if provided few functions that are evaluated during iterations.
///
/// It is merely a skeleton for any iterative optimizer,
/// that I have found to be flexible enough for all my past needs.
/// Here is a simple description of its generic types.
///
/// * `Observation`: the data used as reference during energy evaluations.
/// * `S`: Any type satisfying the `State<Model, E>` trait.
///   Holds that data that is returned after optimization has converged.
/// * `Delta`: Intermediate type, representing the step to be done at each iteration.
///   It can be different from `Model` depending on how you updates your model.
/// * `Model`: The model of what you are trying to optimize.
/// * `PreEval`: Logical first step of computations for the evaluation of this iteration.
/// * `PartialState`: Partially computed new state.
///   Useful to short-circuit the computation of everything needed in the full state
///   in cases where we know that we are going to backtrack
///   (for example if the new energy is higher than the previous one).
/// * `E`: The energy type, typically `f32` or `f64`.
pub trait OptimizerState<Observations, EvalState, Model, Error>
where
    Self: std::marker::Sized,
{
    /// Initialize the optimizer state.
    fn init(obs: &Observations, model: Model) -> Self;

    /// Computes the iteration step from the current optimizer state.
    /// May return `None` if step computation somehow fails.
    /// In such case, iterations are stopped and `iterative` also returns `None`.
    fn step(&self) -> Result<Model, Error>;

    /// Evaluates the model.
    /// You might want to short-circuit evaluation of a full new state depending on your usage.
    /// This is why it returns an `EvalState` and not `S`.
    fn eval(&self, obs: &Observations, new_model: Model) -> EvalState;

    /// Function deciding if iterations should continue.
    /// Also returns the state that will be used for next iteration, or returned if we stop.
    fn stop_criterion(self, nb_iter: usize, eval_state: EvalState) -> (Self, Continue);

    /// Iteratively solve your optimization problem,
    /// with the provided functions by the trait implementation.
    /// Returns the final state `S` and the number of iterations.
    /// May return `None` if a step computation failed (cf `compute_step`).
    fn iterative_solve(obs: &Observations, initial_model: Model) -> Result<(Self, usize), Error> {
        let mut state = Self::init(obs, initial_model);
        let mut nb_iter = 0;
        loop {
            nb_iter = nb_iter + 1;
            let new_model = state.step()?;
            let eval_state = state.eval(obs, new_model);
            let (kept_state, continuation) = state.stop_criterion(nb_iter, eval_state);
            state = kept_state;
            if let Continue::Stop = continuation {
                return Ok((state, nb_iter));
            }
        }
    }
}
