pub enum Continue {
    Stop,
    Forward,
}

pub trait State<Model, E> {
    fn model(&self) -> &Model;
    fn energy(&self) -> E;
}

pub trait Optimizer<Observations, S, Delta, Model, PreEval, PartialState, E>
where
    S: State<Model, E>,
{
    fn initial_energy() -> E;
    fn compute_step(state: &S) -> Option<Delta>;
    fn apply_step(delta: Delta, model: &Model) -> Model;
    fn pre_eval(obs: &Observations, model: &Model) -> PreEval;
    fn eval(obs: &Observations, energy: E, pre_eval: PreEval, model: Model) -> PartialState;
    fn stop_criterion(nb_iter: usize, old_state: S, partial_state: PartialState) -> (S, Continue);

    fn init(obs: &Observations, model: Model) -> PartialState {
        let pre_eval = Self::pre_eval(obs, &model);
        Self::eval(obs, Self::initial_energy(), pre_eval, model)
    }

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
