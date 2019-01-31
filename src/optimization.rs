use std::f32;

pub type Float = f32;

pub enum Continue {
    Stop,
    Forward,
}

pub struct State<Params, Float, Model, Residuals, Jacobian, Gradient> {
    pub params: Params,
    pub energy: Float,
    pub model: Model,
    pub residuals: Residuals,
    pub jacobian: Jacobian,
    pub gradient: Gradient,
}

pub trait QuasiNewtonOptimizer<Observations, P, F, M, R, J, G, Delta>
where
    P: Clone,
{
    fn eval(observations: &Observations, model: M, params: P) -> State<P, F, M, R, J, G>;
    fn step(state: &State<P, F, M, R, J, G>) -> Delta;
    fn apply(delta: Delta, model: &M) -> M;
    fn stop_criterion(
        nb_iter: usize,
        old_state: State<P, F, M, R, J, G>,
        new_state: State<P, F, M, R, J, G>,
    ) -> (State<P, F, M, R, J, G>, Continue);

    fn iterative(
        observations: &Observations,
        mut state: State<P, F, M, R, J, G>,
    ) -> (State<P, F, M, R, J, G>, usize) {
        let mut nb_iter = 0;
        loop {
            nb_iter = nb_iter + 1;
            let delta = Self::step(&state);
            let new_model = Self::apply(delta, &state.model);
            let new_state = Self::eval(observations, new_model, state.params.clone());
            let (kept_state, continuation) = Self::stop_criterion(nb_iter, state, new_state);
            state = kept_state;
            if let Continue::Stop = continuation {
                return (state, nb_iter);
            }
        }
    }
}

pub struct GaussNewtonOptimizer;

// pub struct GaussNewtonOptimizer;
//
// impl QuasiNewtonOptimizer for GaussNewtonOptimizer {
//     eval
// }
//
// GaussNewtonOptimizer::iterative(state)
