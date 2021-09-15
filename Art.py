import tensorflow as tf
import numpy as np
import scipy.misc
import vgg
initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1, dtype=tf.float32)

def content(layer):
  '''
  function to construct content tensors
  '''
  return layer

def styles(layer, type = ""):
  '''
  function to construct style tensors
  '''
  _, height, width, number = map(lambda i: i.value, layer.get_shape())
  size = height * width * number
  feats = tf.reshape(layer, (-1, number))
  gram = tf.matmul(tf.transpose(feats), feats)/size # gram is a matrix
  return gram

def evalFeatureLayers(graph, tensor, myImage):
  # myImage is already preprocessed
  with tf.Session() as sess:
  res = sess.run(tensor,feed_dict={"inputImage:0": myImage})
  return res

def getFeatures(graph, layerNames = ["conv1_1"], featureType = "content", returnTensor
  # contentImage has been normalized
  features = {}
  for layerName in layerNames:
    print ("2",layerName)
    layer = tf.get_default_graph().get_tensor_by_name(layerName+":0")
    if featureType == "content":
      tensor = content(layer)
    elif featureType == "style":
      tensor = styles(layer)
    print (tensor)
    if returnTensor:
      features[layerName] = tensor
    else:
      layerValues = evalFeatureLayers(graph, tensor, inputImage)
      features[layerName] = layerValues
  return features
                
def stylize(contentImage, styleImage,k,j,i, contentLayerNames = ['conv4_2'], styleLayerNames
  imageShape = contentImage.shape
  styleImage = scipy.misc.imresize(styleImage, imageShape)
  contentImage = np.expand_dims(contentImage, axis = 0)
  styleImage = np.expand_dims(styleImage, axis = 0)
  # get the graph
  g = tf.Graph()
  with g.as_default():
    #with tf.Session() as sess:
    # writer = tf.train.SummaryWriter("log_tb",sess.graph)
    image = tf.placeholder('float', shape=(None,imageShape[0],imageShape[1],imageShape
    net, mean_pixel = vgg.net("./models/imagenet-vgg-verydeep-19.mat", image) # input as placeholder
    contentImage = contentImage- mean_pixel
    styleImage = styleImage - mean_pixel
    contentFeatures = getFeatures(g, layerNames = contentLayerNames, featureType
    styleFeatures = getFeatures(g, layerNames = styleLayerNames, featureType = "style"
    print("Phase 1")
  tf.reset_default_graph()
                                
  # get a new graph
  g = tf.Graph()
  with g.as_default():
    image = tf.get_variable("initialImage", shape = (1,)+imageShape,initializer = 
    # reconstruct the tensorflow graph with input as variable
    net, _ = vgg.net("./models/imagenet-vgg-verydeep-19.mat", image)
    contentLoss = 0.0
    styleLoss = 0.0
    for layerName in contentLayerNames:
      contentTensor = getFeatures(g, layerNames = [layerName], featureType = "content"
      contentLoss = contentLoss + tf.nn.l2_loss(contentFeatures[layerName] - contentTensor
    styleLayerWeights = 1.0/len(styleLayerNames)
    for layerName in styleLayerNames:
      styleTensor = getFeatures(g, layerNames = [layerName], featureType = "style"
      styleLoss = styleLoss + styleLayerWeights*tf.nn.l2_loss(styleFeatures[layerName
    # You can modify the loss function
    loss = contentLoss
    train = tf.train.AdamOptimizer(1e1).minimize(loss)
    with tf.Session() as sess:
      sess.run(tf.initialize_all_variables())
      print("Start training")
      for m in range(50):
       sess.run(train)
       print("Training Finished!")
       resultImage = sess.run("initialImage:0")
       resultImage[0,:,:,:] += mean_pixel
   return resultImage
