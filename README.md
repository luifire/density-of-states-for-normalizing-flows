This project contains my master thesis.<br>
Please see thesis.pdf <br><br>

#Abstract:<br>
Normalizing flows are a promising candidate to model complex high-dimensional densities. They can be categorized into volume-preserving (VP) and non-volume preserving (NVP). We investigate in which scenarios one or the other should be preferred.<br><br>
   We examine the difference between VP and NVP normalizing flows by introducing the Density of States (DoS) to the field of deep generative models, which describes the distribution of log density. The ability to change the volume leads to the ability to change the DoS, thus NVP normalizing flows can adjust the DoS, while VP normalizing flows cannot. Matching the DoS is a necessary condition to approximate a distribution well, and so the DoS of the normalizing flow and of the distribution must match. We derive that when a manifold is inflated with noise that matches the prior of the normalizing flow, and the ratio between the manifold dimensionality d and the data dimensionality D is small, the fixed DoS of VP normalizing flows and the DoS of the the inflated manifold can be matched. Thus, for small d/D and prior matching noise, VP and NVP normalizing flows can perform comparable for the approximation of the distribution, and vice versa: when d/D is large or the inflation noise does not match the distribution, VP normalizing flows should not be used.<br><br>
   We experimentally validate our theoretical findings, and find that NVP normalizing flows often perform worse than VPs on the approximation of the distribution of the manifold for small d/D . We investigate the reason for this and find that the term which allows NVP normalizing flow to change the volume is culprit, because it causes a steep loss space that leads to an unstable training, which leads to convergence in bad local minima. 



Note that the normalizing flow was copied and modified from<br>
https://github.com/PolinaKirichenko/flows_ood<br>
<br>
The IWAE is copied from<br>
https://github.com/AntixK/PyTorch-VAE/blob/master/models/iwae.py<br>
<br>
Data processing was adapted from:<br>
https://github.com/didriknielsen/survae_flows<br>
