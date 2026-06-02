
import tensorflow as tf

from tensorflow.keras import layers
import numpy as np

randomSeed = 42


print("TensorFlow version:", tf.__version__)

try:
    register_keras_serializable = tf.keras.saving.register_keras_serializable
    print("Using tf.keras.saving.register_keras_serializable")
except AttributeError:
    register_keras_serializable = tf.keras.utils.register_keras_serializable
    print("Using tf.keras.utils.register_keras_serializable")

try:
    @register_keras_serializable(package="custom", name="swish")
    def swish(x):
        return tf.nn.swish(x)
except Exception as e:
    def swish(x):
        return tf.nn.swish(x)
    print("Warning: register_keras_serializable failed:", e)

# The code below was taken from a post by 
# Sannaraek at https://github.com/getalp/Lightweight-Transformer-Models-For-HAR-on-Mobile-Devices 
# (last accessed 2026-04-02)
# This model has made some modifications, eg. add serializable, build, rename the branch names and parameters.
# BEGIN Copied Code
@register_keras_serializable()
class DropPath(layers.Layer):
    def __init__(self, drop_prob=0.0, **kwargs):
        super(DropPath, self).__init__(**kwargs)
        self.drop_prob = drop_prob

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x,training=None):
        if(training):
            input_shape = tf.shape(x)
            batch_size = input_shape[0]
            rank = x.shape.rank
            shape = (batch_size,) + (1,) * (rank - 1)
            random_tensor = (1 - self.drop_prob) + tf.random.uniform(shape, dtype=x.dtype)
            path_mask = tf.floor(random_tensor)
            output = tf.math.divide(x, 1 - self.drop_prob) * path_mask
            return output
        else:
            return x 

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'drop_prob': self.drop_prob,})
        return config

@register_keras_serializable()
class threeSensorPatches(layers.Layer):
    def __init__(self, projection_dim, patchSize,timeStep, **kwargs):
        super(threeSensorPatches, self).__init__(**kwargs)
        self.patchSize = patchSize
        self.timeStep = timeStep
        self.projection_dim = projection_dim
        self.accProjection = None
        self.magProjection = None
        self.pProjection = None

    def build(self, input_shape):
        super().build(input_shape)
        self.accProjection = layers.Conv1D(filters = int(self.projection_dim//3),kernel_size = self.patchSize,strides = self.timeStep, data_format = "channels_last")
        self.magProjection = layers.Conv1D(filters = int(self.projection_dim//3),kernel_size = self.patchSize,strides = self.timeStep, data_format = "channels_last")
        self.pProjection = layers.Conv1D(filters = int(self.projection_dim//3),kernel_size = self.patchSize,strides = self.timeStep, data_format = "channels_last")

    def call(self, inputData):
        accProjections = self.accProjection(inputData[:,:,:3])
        magProjections = self.magProjection(inputData[:,:,3:6])
        pProjections = self.pProjection(inputData[:,:,6:7])
        Projections = tf.concat((accProjections,magProjections,pProjections),axis=2)
        return Projections
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patchSize': self.patchSize,
            'projection_dim': self.projection_dim,
            'timeStep': self.timeStep,})
        return config

@register_keras_serializable()
class twoSensorPatches(layers.Layer):
    """Patch projection for 4D input (3-axis acc + depth), no magnetometer."""
    def __init__(self, projection_dim, patchSize, timeStep, **kwargs):
        super(twoSensorPatches, self).__init__(**kwargs)
        self.patchSize = patchSize
        self.timeStep = timeStep
        self.projection_dim = projection_dim
        self.accProjection = None
        self.pProjection = None

    def build(self, input_shape):
        super().build(input_shape)
        self.accProjection = layers.Conv1D(filters=int(self.projection_dim // 2), kernel_size=self.patchSize, strides=self.timeStep, data_format="channels_last")
        self.pProjection = layers.Conv1D(filters=int(self.projection_dim // 2), kernel_size=self.patchSize, strides=self.timeStep, data_format="channels_last")

    def call(self, inputData):
        accProjections = self.accProjection(inputData[:, :, :3])
        pProjections = self.pProjection(inputData[:, :, 3:4])
        return tf.concat((accProjections, pProjections), axis=2)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patchSize': self.patchSize,
            'projection_dim': self.projection_dim,
            'timeStep': self.timeStep,})
        return config


@register_keras_serializable()
class ClassToken(layers.Layer):
    def __init__(self, hidden_size,**kwargs):
        super(ClassToken, self).__init__(**kwargs)
        self.cls_init = tf.random.normal
        self.hidden_size = hidden_size
        self.cls = None

    def build(self, input_shape):
        super().build(input_shape)
        self.cls = tf.Variable(
            name="cls",
            initial_value=self.cls_init(shape=(1, 1, self.hidden_size), seed=randomSeed, dtype="float32"),
            trainable=True,
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_broadcasted = tf.cast(
            tf.broadcast_to(self.cls, [batch_size, 1, self.hidden_size]),
            dtype=inputs.dtype,
        )
        return tf.concat([cls_broadcasted, inputs], 1)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'hidden_size': self.hidden_size,})
        return config

@register_keras_serializable()
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim,**kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.position_embedding = None

    def build(self, input_shape):
        super().build(input_shape)
        self.position_embedding = layers.Embedding(input_dim=self.num_patches, output_dim=self.projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = patch + self.position_embedding(positions)
        return encoded
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim,})
        return config

@register_keras_serializable()
class liteFormer(layers.Layer):
    def __init__(self, startIndex, stopIndex, projectionSize, kernelSize=16, attentionHead=3, use_bias=False, dropPathRate=0.0, dropout_rate=0, **kwargs):
        super(liteFormer, self).__init__(**kwargs)
        self.use_bias = use_bias
        self.startIndex = startIndex
        self.stopIndex = stopIndex
        self.kernelSize = kernelSize
        self.softmax = tf.nn.softmax
        self.projectionSize = projectionSize
        self.attentionHead = attentionHead
        self.DropPathLayer = DropPath(dropPathRate)
        self.projectionHalf = projectionSize // 2
        self.depthwise_kernel = None
        self.convBias = None

    def build(self, inputShape):
        super().build(inputShape)
        # Initialize depthwise kernels
        self.depthwise_kernel = [
            self.add_weight(
                shape=(self.kernelSize, 1, 1),
                initializer="glorot_uniform",
                trainable=True,
                name=f"convWeights{_}",
                dtype="float32"
            ) for _ in range(self.attentionHead)
        ]
        # Initialize convolution bias if needed
        if self.use_bias:
            self.convBias = self.add_weight(
                shape=(self.attentionHead,),
                initializer="glorot_uniform",
                trainable=True,
                name="biasWeights",
                dtype="float32"
            )

    def call(self, inputs, training=None):
        # Format inputs
        formattedInputs = inputs[:, :, self.startIndex:self.stopIndex]
        inputShape = tf.shape(formattedInputs)
        reshapedInputs = tf.reshape(formattedInputs, (-1, inputShape[1], self.attentionHead))
        if training:
            for convIndex in range(self.attentionHead):
                self.depthwise_kernel[convIndex].assign(self.softmax(self.depthwise_kernel[convIndex], axis=0))
        convOutputs = [
            tf.nn.conv1d(
                reshapedInputs[:, :, convIndex:convIndex + 1],
                self.depthwise_kernel[convIndex],
                stride=1,
                padding='SAME',
                data_format='NWC' 
            ) for convIndex in range(self.attentionHead)
        ]
        convOutputs = tf.stack(convOutputs, axis=2)
        convOutputsDropPath = self.DropPathLayer(convOutputs)
        localAttention = tf.reshape(convOutputsDropPath, (-1, inputShape[1], self.projectionSize))
        return localAttention

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'use_bias': self.use_bias,
            'kernelSize': self.kernelSize,
            'startIndex': self.startIndex,
            'stopIndex': self.stopIndex,
            'projectionSize': self.projectionSize,
            'attentionHead': self.attentionHead,
        })
        return config

@register_keras_serializable()
class SensorWiseMHA(layers.Layer):
    def __init__(self, projectionQuarter, num_heads,startIndex,stopIndex,dropout_rate = 0.0,dropPathRate = 0.0, **kwargs):
        super(SensorWiseMHA, self).__init__(**kwargs)
        self.projectionQuarter = projectionQuarter
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.MHA = None
        self.startIndex = startIndex
        self.stopIndex = stopIndex
        self.dropPathRate = dropPathRate
        self.DropPath = DropPath(dropPathRate)

    def build(self, input_shape):
        super().build(input_shape)
        self.MHA = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.projectionQuarter, dropout = self.dropout_rate )

    def call(self, inputData, training=None, return_attention_scores = False):
        extractedInput = inputData[:,:,self.startIndex:self.stopIndex]
        if(return_attention_scores):
            MHA_Outputs, attentionScores = self.MHA(extractedInput,extractedInput,return_attention_scores = True )
            return MHA_Outputs , attentionScores
        else:
            MHA_Outputs = self.MHA(extractedInput,extractedInput)
            MHA_Outputs = self.DropPath(MHA_Outputs)
            return MHA_Outputs
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'projectionQuarter': self.projectionQuarter,
            'num_heads': self.num_heads,
            'startIndex': self.startIndex,
            'dropout_rate': self.dropout_rate,
            'stopIndex': self.stopIndex,
            'dropPathRate': self.dropPathRate,})
        return config

def mlp2(x, hidden_units, dropout_rate):
    x = layers.Dense(hidden_units[0],activation=tf.nn.swish)(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(hidden_units[1])(x)
    return x

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.swish)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

@register_keras_serializable()
class L2NormalizeAngles(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        t = tf.reshape(inputs, (-1, 3, 2))
        t = tf.nn.l2_normalize(t, axis=-1)
        return tf.reshape(t, (-1, 6))

def HART(input_shape, output_dim, projection_dim = 192,patchSize = 16,timeStep = 16,num_heads = 3,filterAttentionHead = 4, convKernels = [3, 7, 15, 31, 31, 31], mlp_head_units = [1024],dropout_rate = 0.3,useTokens = False):
    projectionHalf = projection_dim//2
    projectionQuarter = projection_dim//4
    dropPathRate = np.linspace(0, dropout_rate* 10, len(convKernels)) * 0.1
    transformer_units = [
    projection_dim * 2,
    projection_dim,]
    inputs = layers.Input(shape=input_shape)
    # Auto-select patch layer based on input dimensions:
    if input_shape[-1] == 4:
        patches = twoSensorPatches(projection_dim, patchSize, timeStep)(inputs)
    else:
        patches = threeSensorPatches(projection_dim, patchSize, timeStep)(inputs)
    if(useTokens):
        patches = ClassToken(projection_dim)(patches)
    patchCount = patches.shape[1] 
    encoded_patches = PatchEncoder(patchCount, projection_dim)(patches)
    # Create multiple layers of the Transformer block.
    for layerIndex, kernelLength in enumerate(convKernels):        
        x1 = layers.LayerNormalization(epsilon=1e-6 , name = "normalizedInputs_"+str(layerIndex))(encoded_patches)
        branch1 = liteFormer(
                          startIndex = projectionQuarter,
                          stopIndex = projectionQuarter + projectionHalf,
                          projectionSize = projectionHalf,
                          attentionHead =  filterAttentionHead, 
                          kernelSize = kernelLength,
                          dropPathRate = dropPathRate[layerIndex],
                          dropout_rate = dropout_rate,
                          name = "liteFormer_"+str(layerIndex))(x1)

                          
        branch2Acc = SensorWiseMHA(projectionQuarter,num_heads,0,projectionQuarter,dropPathRate = dropPathRate[layerIndex],dropout_rate = dropout_rate,name = "AccMHA_"+str(layerIndex))(x1)

        branch2Meg = SensorWiseMHA(projectionQuarter,num_heads,projectionQuarter + projectionHalf ,projection_dim,dropPathRate = dropPathRate[layerIndex],dropout_rate = dropout_rate, name = "MegMHA_"+str(layerIndex))(x1)
        concatAttention = layers.Concatenate(axis=2)((branch2Acc,branch1,branch2Meg))

        
        x2 = layers.Add()([concatAttention, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp2(x3, hidden_units=transformer_units, dropout_rate=dropout_rate)
        x3 = DropPath(dropPathRate[layerIndex])(x3)
        encoded_patches = layers.Add()([x3, x2])
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    if(useTokens):
        representation = layers.Lambda(lambda v: v[:, 0], name="ExtractToken")(representation)
    else:
        representation = layers.GlobalAveragePooling1D()(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=dropout_rate)
    x0= layers.Dense(output_dim, activation=None)(features)
    x = L2NormalizeAngles()(x0)
    model = tf.keras.Model(inputs=inputs, outputs=x)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        ),
        loss=tf.keras.losses.MeanSquaredError(name="mse_loss"),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.MeanSquaredError(name="mse")
        ]
    )
    return model
# END Copied Code