
# Simple DCGAN

Deep convolutional generative adversarial network implemented in a python class for generating anime character faces.<br>
<img title="dcgan output" alt="unable to preview" src="./gan_outputs.gif">



#### Hyper parameters:

  * Image input shape - [64,64,3]<br>
  * latent dims for generator = 128<br>
  * optimizer: Adam, learning rate = 0.00002, beta1=0.5
  * batch_size: 200
  * number training steps of the discriminator per training step of the generator : 6
### Generator architecture:<br>

<img title="Generator architecture" alt="unable to preview" src="./gen_output.png"><br>

### Discriminator architecture:<br>

<img title="Discriminator architecture" alt="unable to preview" src="./disc_output.png"><br>
