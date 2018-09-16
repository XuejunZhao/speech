# -*- coding: utf-8 -*-
from __future__ import division
from keras.optimizers import Optimizer
from keras.legacy import interfaces
from keras import backend as K
import math
import collections
import numpy
import tensorflow as tf
from six.moves import xrange
# from tensorflow_speech_recognition_challenge.python.per_example_gradients import per_example_gradients

EpsDelta = collections.namedtuple("EpsDelta", ["spent_eps", "spent_delta"])
ClipOption = collections.namedtuple("ClipOption",
                                    ["l2norm_bound", "clip"])
OrderedDict = collections.OrderedDict
# Flags that control privacy spending during training.
tf.flags.DEFINE_integer("batches_per_lot", 1,
                        "Number of batches per lot.")
tf.flags.DEFINE_float("lr", 0.05, "start learning rate")
tf.flags.DEFINE_float("eps", 1.0,
                      "Start privacy spending for one epoch of training, "
                      "used if accountant_type is Amortized.")
tf.flags.DEFINE_float("sigma", 4.0,
                      "Noise sigma, used only if accountant_type is Moments")


def _ListUnion(list_1, list_2):
  """Returns the union of two lists.

  Python sets can have a non-deterministic iteration order. In some
  contexts, this could lead to TensorFlow producing two different
  programs when the same Python script is run twice. In these contexts
  we use lists instead of sets.

  This function is not designed to be especially fast and should only
  be used with small lists.

  Args:
    list_1: A list
    list_2: Another list

  Returns:
    A new list containing one copy of each unique element of list_1 and
    list_2. Uniqueness is determined by "x in union" logic; e.g. two
    string of that value appearing in the union.

  Raises:
    TypeError: The arguments are not lists.
  """

  if not (isinstance(list_1, list) and isinstance(list_2, list)):
    raise TypeError("Arguments must be lists.")

  union = []
  for x in list_1 + list_2:
    if x not in union:
      union.append(x)

  return union


def Interface(ys, xs):
  """Maps xs to consumers.

    Returns a dict mapping each element of xs to any of its consumers that are
    indirectly consumed by ys.

  Args:
    ys: The outputs
    xs: The inputs
  Returns:
    out: Dict mapping each member x of `xs` to a list of all Tensors that are
         direct consumers of x and are eventually consumed by a member of
         `ys`.
  """

  if isinstance(ys, (list, tuple)):
    queue = list(ys)
  else:
    queue = [ys]

  out = OrderedDict()
  if isinstance(xs, (list, tuple)):
    for x in xs:
      out[x] = []
  else:
    out[xs] = []

  done = set()

  while queue:
    y = queue.pop()
    if y in done:
      continue
    done = done.union(set([y]))
    for x in y.op.inputs:
      if x in out:
        out[x].append(y)
      else:
        assert id(x) not in [id(foo) for foo in out]
    queue.extend(y.op.inputs)

  return out


class PXGRegistry(object):
  """Per-Example Gradient registry.

  Maps names of ops to per-example gradient rules for those ops.
  These rules are only needed for ops that directly touch values that
  are shared between examples. For most machine learning applications,
  this means only ops that directly operate on the parameters.


  See http://arxiv.org/abs/1510.01799 for more information, and please
  consider citing that tech report if you use this function in published
  research.
  """

  def __init__(self):
    self.d = OrderedDict()

  def __call__(self, op,
               colocate_gradients_with_ops=False,
               gate_gradients=False):
    if op.node_def.op not in self.d:
      raise NotImplementedError("No per-example gradient rule registered "
                                "for " + op.node_def.op + " in pxg_registry.") #ExpandDims
    return self.d[op.node_def.op](op,
                                  colocate_gradients_with_ops,
                                  gate_gradients)

  def Register(self, op_name, pxg_class):
    """Associates `op_name` key with `pxg_class` value.

    Registers `pxg_class` as the class that will be called to perform
    per-example differentiation through ops with `op_name`.

    Args:
      op_name: String op name.
      pxg_class: An instance of any class with the same signature as MatMulPXG.
    """
    self.d[op_name] = pxg_class


pxg_registry = PXGRegistry()


class MatMulPXG(object):
  """Per-example gradient rule for MatMul op.
  """

  def __init__(self, op,
               colocate_gradients_with_ops=False,
               gate_gradients=False):
    """Construct an instance of the rule for `op`.

    Args:
      op: The Operation to differentiate through.
      colocate_gradients_with_ops: currently unsupported
      gate_gradients: currently unsupported
    """
    assert op.node_def.op == "MatMul"
    self.op = op
    self.colocate_gradients_with_ops = colocate_gradients_with_ops
    self.gate_gradients = gate_gradients

  def __call__(self, x, z_grads):
    """Build the graph for the per-example gradient through the op.

    Assumes that the MatMul was called with a design matrix with examples
    in rows as the first argument and parameters as the second argument.

    Args:
      x: The Tensor to differentiate with respect to. This tensor must
         represent the weights.
      z_grads: The list of gradients on the output of the op.

    Returns:
      x_grads: A Tensor containing the gradient with respect to `x` for
       each example. This is a 3-D tensor, with the first axis corresponding
       to examples and the remaining axes matching the shape of x.
    """
    idx = list(self.op.inputs).index(x)
    assert idx != -1
    assert len(z_grads) == len(self.op.outputs)
    assert idx == 1  # We expect weights to be arg 1
    # We don't expect anyone to per-example differentiate with repsect
    # to anything other than the weights.
    x, _ = self.op.inputs
    z_grads, = z_grads
    x_expanded = tf.expand_dims(x, 2)
    z_grads_expanded = tf.expand_dims(z_grads, 1)
    return tf.multiply(x_expanded, z_grads_expanded)


pxg_registry.Register("MatMul", MatMulPXG)


class Conv2DPXG(object):
  """Per-example gradient rule of Conv2d op.

  Same interface as MatMulPXG.
  """

  def __init__(self, op,
               colocate_gradients_with_ops=False,
               gate_gradients=False):

    assert op.node_def.op == "Conv2D"
    self.op = op
    self.colocate_gradients_with_ops = colocate_gradients_with_ops
    self.gate_gradients = gate_gradients

  def _PxConv2DBuilder(self, input_, w, strides, padding):
    """conv2d run separately per example, to help compute per-example gradients.

    Args:
      input_: tensor containing a minibatch of images / feature maps.
              Shape [batch_size, rows, columns, channels]
      w: convolution kernels. Shape
        [kernel rows, kernel columns, input channels, output channels]
      strides: passed through to regular conv_2d
      padding: passed through to regular conv_2d

    Returns:
      conv: the output of the convolution.
         single tensor, same as what regular conv_2d does
      w_px: a list of batch_size copies of w. each copy was used
          for the corresponding example in the minibatch.
           calling tf.gradients on the copy gives the gradient for just
                  that example.
    """
    input_shape = [int(e) for e in input_.get_shape()]
    batch_size = input_shape[0]
    input_px = [tf.slice(
        input_, [example] + [0] * 3, [1] + input_shape[1:]) for example
                in xrange(batch_size)]
    for input_x in input_px:
      assert int(input_x.get_shape()[0]) == 1
    w_px = [tf.identity(w) for example in xrange(batch_size)]
    conv_px = [tf.nn.conv2d(input_x, w_x,
                            strides=strides,
                            padding=padding)
               for input_x, w_x in zip(input_px, w_px)]
    for conv_x in conv_px:
      num_x = int(conv_x.get_shape()[0])
      assert num_x == 1, num_x
    assert len(conv_px) == batch_size
    conv = tf.concat(axis=0, values=conv_px)
    assert int(conv.get_shape()[0]) == batch_size
    return conv, w_px

  def __call__(self, w, z_grads):
    idx = list(self.op.inputs).index(w)
    # Make sure that `op` was actually applied to `w`
    assert idx != -1
    assert len(z_grads) == len(self.op.outputs)
    # The following assert may be removed when we are ready to use this
    # for general purpose code.
    # This assert is only expected to hold in the contex of our preliminary
    # MNIST experiments.
    assert idx == 1  # We expect convolution weights to be arg 1

    images, filters = self.op.inputs
    strides = self.op.get_attr("strides")
    padding = self.op.get_attr("padding")
    # Currently assuming that one specifies at most these four arguments and
    # that all other arguments to conv2d are set to default.

    conv, w_px = self._PxConv2DBuilder(images, filters, strides, padding)
    z_grads, = z_grads

    gradients_list = tf.gradients(conv, w_px, z_grads,
                                  colocate_gradients_with_ops=
                                  self.colocate_gradients_with_ops,
                                  gate_gradients=self.gate_gradients)

    return tf.stack(gradients_list)

pxg_registry.Register("Conv2D", Conv2DPXG)


class AddPXG(object):
  """Per-example gradient rule for Add op.

  Same interface as MatMulPXG.
  """

  def __init__(self, op,
               colocate_gradients_with_ops=False,
               gate_gradients=False):

    assert op.node_def.op == "Add"
    self.op = op
    self.colocate_gradients_with_ops = colocate_gradients_with_ops
    self.gate_gradients = gate_gradients

  def __call__(self, x, z_grads):
    idx = list(self.op.inputs).index(x)
    # Make sure that `op` was actually applied to `x`
    assert idx != -1
    assert len(z_grads) == len(self.op.outputs)
    # The following assert may be removed when we are ready to use this
    # for general purpose code.
    # This assert is only expected to hold in the contex of our preliminary
    # MNIST experiments.
    assert idx == 1 # We expect biases to be arg 1
    # We don't expect anyone to per-example differentiate with respect
    # to anything other than the biases.
    x, _ = self.op.inputs
    z_grads, = z_grads
    return z_grads


pxg_registry.Register("Add", AddPXG)


def PerExampleGradients(ys, xs, grad_ys=None, name="gradients",
                        colocate_gradients_with_ops=False,
                        gate_gradients=False):
  """Symbolic differentiation, separately for each example.

  Matches the interface of tf.gradients, but the return values each have an
  additional axis corresponding to the examples.

  Assumes that the cost in `ys` is additive across examples.
  e.g., no batch normalization.
  Individual rules for each op specify their own assumptions about how
  examples are put into tensors.
  """

  # Find the interface between the xs and the cost
  for x in xs:
    assert isinstance(x, tf.Tensor), type(x)
  interface = Interface(ys, xs)
  merged_interface = []
  for x in xs:
    merged_interface = _ListUnion(merged_interface, interface[x])
  # Differentiate with respect to the interface
  interface_gradients = tf.gradients(ys, merged_interface, grad_ys=grad_ys,
                                     name=name,
                                     colocate_gradients_with_ops=
                                     colocate_gradients_with_ops,
                                     gate_gradients=gate_gradients)
  grad_dict = OrderedDict(zip(merged_interface, interface_gradients))
  # Build the per-example gradients with respect to the xs
  if colocate_gradients_with_ops:
    raise NotImplementedError("The per-example gradients are not yet "
                              "colocated with ops.")
  if gate_gradients:
    raise NotImplementedError("The per-example gradients are not yet "
                              "gated.")
  out = []
  for x in xs:
    zs = interface[x]
    ops = []
    for z in zs:
      ops = _ListUnion(ops, [z.op])
    if len(ops) != 1:
      raise NotImplementedError("Currently we only support the case "
                                "where each x is consumed by exactly "
                                "one op. but %s is consumed by %d ops."
                                % (x.name, len(ops)))
    op = ops[0]
    pxg_rule = pxg_registry(op, colocate_gradients_with_ops, gate_gradients)
    x_grad = pxg_rule(x, [grad_dict[z] for z in zs])
    out.append(x_grad)
  return out


    # def get_gradients(self, loss, params):
    #     grads = K.gradients(loss, params)
    #     if None in grads:
    #         raise ValueError('An operation has `None` for gradient. '
    #                          'Please make sure that all of your ops have a '
    #                          'gradient defined (i.e. are differentiable). '
    #                          'Common ops without gradient: '
    #                          'K.argmax, K.round, K.eval.')
    #     if hasattr(self, 'clipnorm') and self.clipnorm > 0:
    #         norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
    #         grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
    #     if hasattr(self, 'clipvalue') and self.clipvalue > 0:
    #         grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
    #     return grads

class AmortizedAccountant(object):
  """Keep track of privacy spending in an amortized way.

  AmortizedAccountant accumulates the privacy spending by assuming
  all the examples are processed uniformly at random so the spending is
  amortized among all the examples. And we assume that we use Gaussian noise
  so the accumulation is on eps^2 and delta, using advanced composition.
  """

  def __init__(self, total_examples):
    """Initialization. Currently only support amortized tracking.

    Args:
      total_examples: total number of examples.
    """

    assert total_examples > 0
    self._total_examples = total_examples
    self._eps_squared_sum = tf.Variable(tf.zeros([1]), trainable=False,
                                        name="eps_squared_sum")
    self._delta_sum = tf.Variable(tf.zeros([1]), trainable=False,
                                  name="delta_sum")

  def accumulate_privacy_spending(self, eps_delta, unused_sigma,
                                  num_examples):
    """Accumulate the privacy spending.

    Currently only support approximate privacy. Here we assume we use Gaussian
    noise on randomly sampled batch so we get better composition: 1. the per
    batch privacy is computed using privacy amplication via sampling bound;
    2. the composition is done using the composition with Gaussian noise.
    TODO(liqzhang) Add a link to a document that describes the bounds used.

    Args:
      eps_delta: EpsDelta pair which can be tensors.
      unused_sigma: the noise sigma. Unused for this accountant.
      num_examples: the number of examples involved.
    Returns:
      a TensorFlow operation for updating the privacy spending.
    """

    eps, delta = eps_delta
    with tf.control_dependencies(
        [tf.Assert(tf.greater(delta, 0),
                   ["delta needs to be greater than 0"])]):
      amortize_ratio = (tf.cast(num_examples, tf.float32) * 1.0 /
                        self._total_examples)
      # Use privacy amplification via sampling bound.
      # See Lemma 2.2 in http://arxiv.org/pdf/1405.7085v2.pdf
      # TODO(liqzhang) Add a link to a document with formal statement
      # and proof.
      amortize_eps = tf.reshape(tf.log(1.0 + amortize_ratio * (
          tf.exp(eps) - 1.0)), [1])
      amortize_delta = tf.reshape(amortize_ratio * delta, [1])
      return tf.group(*[tf.assign_add(self._eps_squared_sum,
                                      tf.square(amortize_eps)),
                        tf.assign_add(self._delta_sum, amortize_delta)])
      
  def get_privacy_spent(self, sess, target_eps=None):
    """Report the spending so far.

    Args:
      sess: the session to run the tensor.
      target_eps: the target epsilon. Unused.
    Returns:
      the list containing a single EpsDelta, with values as Python floats (as
      opposed to numpy.float64). This is to be consistent with
      MomentAccountant which can return a list of (eps, delta) pair.
    """

    # pylint: disable=unused-argument
    unused_target_eps = target_eps
    eps_squared_sum, delta_sum = sess.run([self._eps_squared_sum,
                                           self._delta_sum])
    return [EpsDelta(math.sqrt(eps_squared_sum), float(delta_sum))]

def Interface(ys, xs):
  """Maps xs to consumers.

    Returns a dict mapping each element of xs to any of its consumers that are
    indirectly consumed by ys.

  Args:
    ys: The outputs
    xs: The inputs
  Returns:
    out: Dict mapping each member x of `xs` to a list of all Tensors that are
         direct consumers of x and are eventually consumed by a member of
         `ys`.
  """

  if isinstance(ys, (list, tuple)):
    queue = list(ys)
  else:
    queue = [ys]

  out = OrderedDict()
  if isinstance(xs, (list, tuple)):
    for x in xs:
      out[x] = []
  else:
    out[xs] = []

  done = set()

  while queue:
    y = queue.pop()
    if y in done:
      continue
    done = done.union(set([y]))
    for x in y.op.inputs:
      if x in out:
        out[x].append(y)
      else:
        assert id(x) not in [id(foo) for foo in out]
    queue.extend(y.op.inputs)

  return out

# def GetTensorOpName(x):
# 	  """Get the name of the op that created a tensor.

# 	  Useful for naming related tensors, as ':' in name field of op is not permitted

# 	  Args:
# 	    x: the input tensor.
# 	  Returns:
# 	    the name of the op.
# 	  """
# 	t = x.name.split(":", 1)
# 	if len(t) == 1:
# 		return x.name
# 	else:
# 		return t[0]

def AddGaussianNoise(t, sigma, name=None):
  """Add i.i.d. Gaussian noise (0, sigma^2) to every entry of t.

  Args:
    t: the input tensor.
    sigma: the stddev of the Gaussian noise.
    name: optional name.
  Returns:
    the noisy tensor.
  """

  with tf.name_scope(values=[t, sigma], name=name,
                     default_name="add_gaussian_noise") as name:
    noisy_t = t + tf.random_normal(tf.shape(t), stddev=sigma)
  return noisy_t

# class AmortizedAccountant(object):
#   """Keep track of privacy spending in an amortized way.

#   AmortizedAccountant accumulates the privacy spending by assuming
#   all the examples are processed uniformly at random so the spending is
#   amortized among all the examples. And we assume that we use Gaussian noise
#   so the accumulation is on eps^2 and delta, using advanced composition.
#   """

#   def __init__(self, total_examples):
#     """Initialization. Currently only support amortized tracking.

#     Args:
#       total_examples: total number of examples.
#     """
# 	assert total_examples > 0
#     self._total_examples = total_examples
#     self._eps_squared_sum = tf.Variable(tf.zeros([1]), trainable=False,
#                                         name="eps_squared_sum")
#     self._delta_sum = tf.Variable(tf.zeros([1]), trainable=False,
#                                   name="delta_sum")

def sanitize(x, eps_delta, sigma=None,
               tensor_name=None,
               num_examples=None, add_noise=True):
    """Sanitize the given tensor.

    This santize a given tensor by first applying l2 norm clipping and then
    adding Gaussian noise. It calls the privacy accountant for updating the
    privacy spending.

    Args:
      x: the tensor to sanitize.
      eps_delta: a pair of eps, delta for (eps,delta)-DP. Use it to
        compute sigma if sigma is None.
      sigma: if sigma is not None, use sigma.
      option: a ClipOption which, if supplied, used for
        clipping and adding noise.
      tensor_name: the name of the tensor.
      num_examples: if None, use the number of "rows" of x.
      add_noise: if True, then add noise, else just clip.
    Returns:
      a pair of sanitized tensor and the operation to accumulate privacy
      spending.
    """

    if sigma is None:
      eps, delta = eps_delta
        # The following formula is taken from
        #   Dwork and Roth, The Algorithmic Foundations of Differential
        #   Privacy, Appendix A.
        #   http://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
    sigma = tf.sqrt(2.0 * tf.log(1.25 / delta)) / eps

    if add_noise:
      if num_examples is None:
        num_examples = tf.slice(tf.shape(x), [0], [1])
      # high moments
      privacy_accum_op = self._accountant.accumulate_privacy_spending(
          eps_delta, sigma, num_examples)
      with tf.control_dependencies([privacy_accum_op]):
        saned_x = AddGaussianNoise(tf.reduce_sum(x, 0),
                                         sigma * l2norm_bound)
    else:
      saned_x = tf.reduce_sum(x, 0)
    return saned_x

class SGD_DP(Optimizer):
    """Stochastic gradient descent optimizer.
    Includes support for momentum,
    learning rate decay, and Nesterov momentum.
    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, **kwargs):
        super(SGD_DP, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr') #define
            self.momentum = K.variable(momentum, name='momentum') #define
            self.decay = K.variable(decay, name='decay') #define
        self.initial_decay = decay
        self.nesterov = nesterov #define
        #define change status

    def compute_sanitized_gradients(self, loss, var_list=None,
                                  add_noise=True):
	    """Compute the sanitized gradients.
	    Args:
	      loss: the loss tensor.
	      var_list: the optional variables.
	      add_noise: if true, then add noise. Always clip.
	    Returns:
	      a pair of (list of sanitized gradients) and privacy spending accumulation
	      operations.
	    Raises:
	      TypeError: if var_list contains non-variable.
	       self._assert_valid_dtypes([loss])
	    """
	    xs = [tf.convert_to_tensor(x) for x in var_list]
	    px_grads = PerExampleGradients(loss, xs)#per_example_gradients
	    sanitized_grads = []
	    for px_grad, v in zip(px_grads, var_list):

	      # tensor_name = GetTensorOpName(v)
	      #tensorname
	      #tensor_name=tensor_name,
	        sanitized_grad = self._sanitizer.sanitize(
		          px_grad, self._eps_delta, sigma=self._sigma,
		          add_noise=add_noise,
		          num_examples=self._batches_per_lot * tf.slice(
	              tf.shape(px_grad), [0], [1]))
	        sanitized_grads.append(sanitized_grad)
	    return sanitized_grads


    #perSGD function test 与perSGD的效果
    #loss type
    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
      grads = self.get_gradients(loss, params) # change 
      # var_list = tf.trainable_variables()
      # grads = self.compute_sanitized_gradients(loss, var_list)
      # sanitized_grads = []
      # for px_grad, v in zip(px_grads, var_list):

      #   # tensor_name = GetTensorOpName(v)
      #   #tensorname
      #   #tensor_name=tensor_name,
      #     sanitized_grad = self._sanitizer.sanitize(
      #         px_grad, self._eps_delta, sigma=self._sigma,
      #         add_noise=add_noise,
      #         num_examples=self._batches_per_lot * tf.slice(
      #           tf.shape(px_grad), [0], [1]))
      #     sanitized_grads.append(sanitized_grad)

      self.updates = [K.update_add(self.iterations, 1)]
      lr = self.lr
      if self.initial_decay > 0:
          lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                    K.dtype(self.decay))))
      # momentum
      shapes = [K.int_shape(p) for p in params]
      moments = [K.zeros(shape) for shape in shapes]
      self.weights = [self.iterations] + moments
      for p, g, m in zip(params, grads, moments):
          # g = sanitize(g,self._eps_delta, sigma=self._sigma,
          #     add_noise=add_noise,
          #     num_examples=self._batches_per_lot * tf.slice(
          #       tf.shape(px_grad), [0], [1]))

          v = self.momentum * m - lr * g  # velocity
          self.updates.append(K.update(m, v))

          if self.nesterov:
              new_p = p + self.momentum * v - lr * g
          else:
              new_p = p + v

          # Apply constraints.
          if getattr(p, 'constraint', None) is not None:
              new_p = p.constraint(new_p)

          self.updates.append(K.update(p, new_p))
      return self.updates


    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov}
        base_config = super(SGD_DP, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
