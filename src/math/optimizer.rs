//! Guiding traits to implement iterative optimization algorithms.

/// Enum used to indicate if iterations should continue or stop.
/// Must be returned by the stop_criterion function.
pub enum Continue {
    /// Stop iterations.
    Stop,
    /// Continue iterations.
    Forward,
}

/// A `State<Model, E>` must provide access to its encapsulated model and energy.
pub trait State<Model, E> {
    /// Retrieve the model in the state.
    fn model(&self) -> &Model;
    /// Retrieve the energy in the state.
    fn energy(&self) -> E;
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
pub trait Optimizer<Observations, S, Delta, Model, PreEval, PartialState, E>
where
    S: State<Model, E>,
{
    /// Provides the energy at initialization of the optimizer,
    /// typically `f32::INFINITY` or similar.
    fn initial_energy() -> E;

    /// Computes the iteration step from the current optimizer state.
    /// May return `None` if step computation somehow fails.
    /// In such case, iterations are stopped and `iterative` also returns `None`.
    fn compute_step(state: &S) -> Option<Delta>;

    /// Updates the model from the computed Delta.
    fn apply_step(delta: Delta, model: &Model) -> Model;

    /// From observations and a model, computes preliminary data useful for model evaluation.
    fn pre_eval(obs: &Observations, model: &Model) -> PreEval;

    /// Evaluates the model.
    /// You might want to short-circuit evaluation of a full new state depending on your usage.
    /// This can be indicated by the dedicated `PartialState` return type.
    fn eval(obs: &Observations, energy: E, pre_eval: PreEval, model: Model) -> PartialState;

    /// Function deciding if iterations should continue.
    /// Also returns the state that will be used for next iteration, or returned if we stop.
    fn stop_criterion(nb_iter: usize, old_state: S, partial_state: PartialState) -> (S, Continue);

    /// Helper to initialize your `Optimizer`.
    /// You than just have to convert this `PartialState` into a state `S`
    /// and you are ready to call `iterative`.
    fn init(obs: &Observations, model: Model) -> PartialState {
        let pre_eval = Self::pre_eval(obs, &model);
        Self::eval(obs, Self::initial_energy(), pre_eval, model)
    }

    /// Iteratively solve your optimization problem,
    /// with the provided functions by the trait implementation.
    /// Returns the final state `S` and the number of iterations.
    /// May return `None` if a step computation failed (cf `compute_step`).
    fn iterative(obs: &Observations, mut state: S) -> Option<(S, usize)> {
        let mut nb_iter = 0;
        loop {
            nb_iter = nb_iter + 1;
            match Self::compute_step(&state) {
                Some(delta) => {
                    let new_model = Self::apply_step(delta, state.model());
                    let pre_eval = Self::pre_eval(obs, &new_model);
                    let partial_new_state = Self::eval(obs, state.energy(), pre_eval, new_model);
                    let (kept_state, continuation) =
                        Self::stop_criterion(nb_iter, state, partial_new_state);
                    state = kept_state;
                    if let Continue::Stop = continuation {
                        return Some((state, nb_iter));
                    }
                }
                None => return None,
            }
        }
    }
}
