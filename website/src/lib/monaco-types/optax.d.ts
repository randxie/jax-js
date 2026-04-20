import { DType, JsTree, numpy } from "@jax-js/jax";

//#region src/base.d.ts
/** The internal state of the optimizer, varies in shape. */
type OptState = JsTree<numpy.Array>;
/**
 * A pair of pure functions implementing a gradient transformation.
 *
 * Optimizers are implemented with this interface. They do not contain any
 * internal state. The "optimizer state" `OptState` is initialized, then passed
 * into each update call, which returns a new state.
 *
 * Gradients are transformed during the update call, and they should have the
 * same PyTree shape as the parameters.
 */
interface GradientTransformation {
  init<Params extends JsTree<numpy.Array>>(params: Params): OptState;
  update<Params extends JsTree<numpy.Array>>(updates: Params, state: OptState, params?: Params): [Params, OptState];
}
/** @inline */
type Schedule = (count: number) => number;
/** @inline */
type ScalarOrSchedule = number | Schedule;
/** Simplest possible state for a transformation. */

/** Stateless identity transformation that leaves input gradients untouched. */
declare function identity(): GradientTransformation;
/** Stateless transformation that maps input gradients to zero. */
declare function setToZero(): GradientTransformation;
/**
 * Applies an update to the corresponding parameters.
 *
 * This function is provided for convenience, as it just adds the updates to the
 * parameters directly. You can get updates from a `GradientTransformation`.
 */
declare function applyUpdates<Params extends JsTree<numpy.Array>>(params: Params, updates: Params): Params;
//#endregion
//#region src/transform.d.ts
type ScaleByAdamOptions = {
  b1?: number;
  b2?: number;
  eps?: number;
  epsRoot?: number;
  nesterov?: boolean;
};
declare function scaleByAdam({
  b1,
  b2,
  eps,
  epsRoot,
  nesterov
}?: ScaleByAdamOptions): GradientTransformation;
/** Scale by a constant step size. */
declare function scale(stepSize: number): GradientTransformation;
/** Scale updates using a custom schedule for the step size. */
declare function scaleBySchedule(stepSizeFn: Schedule): GradientTransformation;
/** Scale by the (negative) learning rate (either as scalar or as schedule). */
declare function scaleByLearningRate(learningRate: ScalarOrSchedule, flipSign?: boolean): GradientTransformation;
/** Clip gradients by global norm. */
declare function clipByGlobalNorm(maxNorm: number): GradientTransformation;
type MaskFn = (tree: JsTree<numpy.Array>) => JsTree<numpy.Array>;
type AddDecayedWeightsOptions = {
  weightDecay?: ScalarOrSchedule;
  mask?: JsTree<numpy.Array> | MaskFn | null;
};
/** Add parameter scaled by weight decay. */
declare function addDecayedWeights({
  weightDecay,
  mask
}?: AddDecayedWeightsOptions): GradientTransformation;
type TraceOptions = {
  decay?: number;
  nesterov?: boolean;
};
/** Compute a trace of past updates. */
declare function trace({
  decay,
  nesterov
}?: TraceOptions): GradientTransformation;
//#endregion
//#region src/alias.d.ts
type SgdOptions = {
  momentum?: number | null;
  nesterov?: boolean;
};
/** Stochastic gradient descent. */
declare function sgd(learningRate: ScalarOrSchedule, opts?: SgdOptions): GradientTransformation;
/** The Adam optimizer. */
declare function adam(learningRate: ScalarOrSchedule, opts?: ScaleByAdamOptions): GradientTransformation;
type AdamWOptions = ScaleByAdamOptions & AddDecayedWeightsOptions;
/** Adam with weight decay regularization. */
declare function adamw(learningRate: ScalarOrSchedule, opts?: AdamWOptions): GradientTransformation;
//#endregion
//#region src/combine.d.ts
/** Applies a list of chainable update transformations. */
declare function chain(...transforms: GradientTransformation[]): GradientTransformation;
//#endregion
//#region src/losses.d.ts
/**
 * Calculates squared error for a set of predictions.
 *
 * Mean squared error can be computed as `squaredError(a, b).mean()`.
 */
declare function squaredError(predictions: numpy.Array, targets?: numpy.Array): numpy.Array;
/**
 * Calculates the L2 loss for a set of predictions.
 *
 * This is equivalent to 0.5 * squared error, where the constant is standard
 * from "Pattern Recognition and Machine Learning" by Bishop.
 */
declare function l2Loss(predictions: numpy.Array, targets?: numpy.Array): numpy.Array;
//#endregion
//#region src/treeUtils.d.ts
declare function treeZerosLike(tr: JsTree<numpy.Array>, dtype?: DType): JsTree<numpy.Array>;
declare function treeOnesLike(tr: JsTree<numpy.Array>, dtype?: DType): JsTree<numpy.Array>;
declare function treeUpdateMoment(updates: JsTree<numpy.Array>, moments: JsTree<numpy.Array>, decay: number, order: number): JsTree<numpy.Array>;
/** Performs bias correction, dividing by 1-decay^count. */
declare function treeBiasCorrection(moments: JsTree<numpy.Array>, decay: number, count: numpy.Array): JsTree<numpy.Array>;
/** Sum all elements across all arrays in a pytree. */
declare function treeSum(tr: JsTree<numpy.Array>): numpy.Array;
/** Max of all elements across all arrays in a pytree. */
declare function treeMax(tr: JsTree<numpy.Array>): numpy.Array;
type NormOrd = 1 | 2 | "inf" | "infinity" | number | null;
/** Compute the vector norm of the given ord of a pytree. */
declare function treeNorm(tr: JsTree<numpy.Array>, ord?: NormOrd, squared?: boolean): numpy.Array;
//#endregion
export { type AdamWOptions, type AddDecayedWeightsOptions, type GradientTransformation, type NormOrd, type OptState, type ScaleByAdamOptions, type SgdOptions, type TraceOptions, adam, adamw, addDecayedWeights, applyUpdates, chain, clipByGlobalNorm, identity, l2Loss, scale, scaleByAdam, scaleByLearningRate, scaleBySchedule, setToZero, sgd, squaredError, trace, treeBiasCorrection, treeMax, treeNorm, treeOnesLike, treeSum, treeUpdateMoment, treeZerosLike };