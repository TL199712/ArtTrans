from Art import *
import scipy.misc
import matplotlib.pyplot as plt
from IPython.display import Image
for k in range (1,2):
  for j in range (1,4):
    content_path ='./pics/content_' + str (k) + '.jpg'
    style_path = './pics/style_' + str (j) + '.jpg'
    contentImage = scipy.misc.imread(content_path).astype(np.float)
    styleImage = scipy.misc.imread(style_path).astype(np.float)
    plt.imshow(contentImage/256)
    plt.show()
    plt.imshow(styleImage/256)
    plt.show()
    #contentLayerNames = ['relu4_2']
    #styleLayerNames = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
    #######################################################
    for i in range (1,5):
      contentLayerNames = ['conv' + str(i) + '_1']
      #contentLayerNames = ['conv1_1']
      styleLayerNames = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'
      resultImage = stylize(contentImage, styleImage,k,j,i, contentLayerNames = 
      resultImage = resultImage[0,:,:,:]
      img = np.clip(resultImage, 0, 255).astype(np.uint8)
      output_path = './outputs/content' + str(k) + '_style' + str(j) + '_CLconv'
      #scipy.misc.imsave("./outputs/content1_style1_CLconv11_SLconv11conv21conv31conv41conv51.jpg", img)
      scipy.misc.imsave(output_path, img)
      plt.imshow(img)
      plt.show()
