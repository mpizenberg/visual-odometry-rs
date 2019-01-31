use std::f32;

pub type Float = f32;

// An interative optimizer where each part of the algorithm is personalizable.
pub fn iterative<
    Params,       // Parameters of optimizer, such as \lambda of Levenberg Marquardt
    Observations, // The observations, i.e. the data
    Model,        // Parametrization of the modelization such as a struct
    Residuals,    // The differences between the projection of the model and the observed data
    Jacobian,     // Jacobian of the residuals, typically an NxM matrix (N data, M variables)
    Gradient,     // Typically the matrix: Jacobian^T * Residuals
    EvalFn,       // Function evaluating the jacobian and residuals
    StepFn,       // Function computing the iteration step
    CriterionFn,  // Stopping criterion
>(
    eval: EvalFn,
    step: StepFn,
    stop_criterion: CriterionFn,
    observations: &Observations,
    initial_model: Model,
    initial_params: Params,
) -> (Model, usize)
where
    EvalFn: Fn(&Observations, &Model) -> (Jacobian, Gradient, Residuals, Float),
    StepFn: Fn(&Jacobian, &Residuals, &Model, &Params) -> (Model, Params),
    CriterionFn: Fn(
        usize,
        (Params, Float, Gradient, Model),
        (Params, Float, Gradient, Model),
    ) -> ((Params, Float, Gradient, Model), Continue),
{
    // Evaluate the model for the first time.
    let (jacobian, gradient, residuals, energy) = eval(observations, &initial_model);
    let mut state = (initial_params, energy, gradient, initial_model);
    let mut nb_iter = 0;

    loop {
        nb_iter = nb_iter + 1;
        let (model, params) = step(&jacobian, &residuals, &state.3, &state.0);
        let (jacobian, gradient, residuals, energy) = eval(observations, &model);
        let step_state = (params, energy, gradient, model);
        let (kept_state, continuation) = stop_criterion(nb_iter, state, step_state);
    }
}

pub enum Continue {
    Stop,
    Forward,
}
