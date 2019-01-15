use std::f32;

pub type Float = f32;

pub fn iterative<Observation, Model, Derivatives, Residual, EvalFn, StepFn, CriterionFn>(
    eval: EvalFn,
    step: StepFn,
    stop_criterion: CriterionFn,
    observation: &Observation,
    initial_model: Model,
) -> (Model, usize)
where
    EvalFn: Fn(&Observation, &Model) -> (Derivatives, Residual),
    StepFn: Fn(&Derivatives, &Residual, &Model) -> Model,
    CriterionFn: Fn(usize, Float, &Residual) -> (Float, Continue),
{
    // Manual first iteration enable avoiding to have Clone for model.
    // Otherwise, the compiler doesn't know if previous_model has
    // been initialized in the backward branch.
    let mut energy = f32::INFINITY;
    let (mut derivatives, mut residual) = eval(observation, &initial_model);
    match stop_criterion(0, energy, &residual) {
        (new_energy, Continue::Forward) => {
            energy = new_energy;
        }
        _ => return (initial_model, 0),
    }
    let mut nb_iter = 1;
    let mut model = step(&derivatives, &residual, &initial_model);
    let mut previous_model = initial_model;
    let (new_derivatives, new_residual) = eval(observation, &model);
    derivatives = new_derivatives;
    residual = new_residual;

    // After first iteration, loop until stop criterion.
    loop {
        let (new_energy, continuation) = stop_criterion(nb_iter, energy, &residual);
        match continuation {
            Continue::Stop => break,
            Continue::Backward => {
                model = previous_model;
                break;
            }
            Continue::Forward => {
                nb_iter = nb_iter + 1;
                energy = new_energy;
                previous_model = model;
                model = step(&derivatives, &residual, &previous_model);
                let (new_derivatives, new_residual) = eval(observation, &model);
                derivatives = new_derivatives;
                residual = new_residual;
            }
        }
    }
    (model, nb_iter)
}

pub enum Continue {
    Stop,
    Forward,
    Backward,
}
