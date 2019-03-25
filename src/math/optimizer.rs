// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Guiding traits to implement iterative optimization algorithms.

/// Enum used to indicate if iterations should continue or stop.
/// Must be returned by the `stop_criterion` function.
pub enum Continue {
    /// Stop iterations.
    Stop,
    /// Continue iterations.
    Forward,
}

/// An `State<Observations, EvalState, Model, Error>`
/// is capable of iteratively minimizing an energy function,
/// if provided few functions that are evaluated during iterations.
///
/// It merely is a skeleton for any iterative optimizer,
/// that I have found to be flexible enough for all my past needs.
/// Here is a simple description of its generic types.
///
/// * `Self`: State of the iterative optimizer.
/// * `Observations`: Data used as reference during energy evaluations.
/// * `EvalState`: Data computed while evaluating a model just computed.
///   Will typically be result successfully containing all the data
///   needed to update the optimizer state,
///   or an error meaning that we stopped the evaluation because the energy increased.
/// * `Model`: The model of what you are trying to optimize.
/// * `Error`: Custom error type for potential failures in step computation.
pub trait State<Observations, EvalState, Model, Error>
where
    Self: std::marker::Sized,
{
    /// Initialize the optimizer state.
    fn init(obs: &Observations, model: Model) -> Self;

    /// Compute an iteration step from the current optimizer state.
    /// May fail, in such cases, iterations are stopped.
    fn step(&self) -> Result<Model, Error>;

    /// Evaluate the model.
    /// You might want to short-circuit evaluation of a full new state depending on your usage
    /// (e.g. if the energy increases).
    /// This is why it returns an `EvalState` and not `Self`.
    fn eval(&self, obs: &Observations, new_model: Model) -> EvalState;

    /// Function deciding if iterations should continue.
    /// Also return the state that will be used for next iteration.
    fn stop_criterion(self, nb_iter: usize, eval_state: EvalState) -> (Self, Continue);

    /// Iteratively solve your optimization problem,
    /// with the provided functions by the trait implementation.
    /// Return the final state and the number of iterations.
    /// May return an error if a step computation failed.
    fn iterative_solve(obs: &Observations, initial_model: Model) -> Result<(Self, usize), Error> {
        let mut state = Self::init(obs, initial_model);
        let mut nb_iter = 0;
        loop {
            nb_iter += 1;
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
