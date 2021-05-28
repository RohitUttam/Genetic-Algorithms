
<!-- Author: Rohit Uttamchandani -->
<!-- Title: Introduction to Genetic Algorithms -->
<!-- Date: 28.05.2021 -->
<dl>
<br>
<br>
<center><b><span style="font-size:3em"> Introduction to Genetic Algorithms</b></center>
<hr style="height:2px;border-width:0;color:#bfc6c7;background-color:#bfc6c7;width:70%"></hr>
<br>
<a id='top'></a>
<br>
<br>
<a href='#Motivation'> 1. Motivation </a> 
<br>
<br>
<a href='#Intro'> 2. Introduction to optimization </a>  
<p style="margin-left: 40px"><a href='#globallocal'> 2.1. Local vs. global optima  </a> </p>
<p style="margin-left: 40px"><a href='#searchspace'> 2.2. Search space </a> </p>
<p style="margin-left: 40px"><a href='#solutionspace'> 2.3. Solution space </a> </p>
<p style="margin-left: 40px"><a href='#ambiguityredundancy'> 2.4. Ambiguity vs. Redundancy </a> </p>
<p style="margin-left: 40px"><a href='#explorationexploitation'> 2.5. Exploration vs. Exploitation </a> </p>
<br>
<a href='#Methods'> 3. Exact vs. approximate methods </a> 
<br>
<br>
<a href='#GeneticAlgorithms'> 4. Genetic Algorithms </a> 
<p style="margin-left: 40px"><a href='#Basics'> 4.1. Basic concepts </a> </p>
<p style="margin-left: 40px"><a href='#Lifecycle'> 4.2. Evolutionary lifecycle </a> </p>
<p style="margin-left: 40px"><a href='#ProsCons'> 4.3. Pros and Cons </a> </p>
<br>
<a href='#Game'> 5. Example: Target Sum Game </a>
<p style="margin-left: 40px"><a href='#Encoding'> 5.1. Encoding </a> </p>
<p style="margin-left: 40px"><a href='#Initialization'> 5.2. Initialization </a> </p>
<p style="margin-left: 40px"><a href='#Fitness'> 5.3. Fitness </a> </p>
<p style="margin-left: 40px"><a href='#Selection'> 5.4. Selection </a> </p>
<p style="margin-left: 40px"><a href='#Crossover'> 5.5. Crossover </a> </p>
<p style="margin-left: 40px"><a href='#Mutation'> 5.6. Mutation </a> </p>
<p style="margin-left: 40px"><a href='#Replacement'> 5.7. Replacement </a> </p>
<p>
<br>
<a href='#Biblio'> Bibliography </a>
<br>

<br>
</p>



<br>
<br>
<br>
<a id='Motivation'></a>
<p style="margin-left: 5px; font-size: 1.4em"> <b> 1. Motivation</b></p>
<hr style="height:1.5px;border-width:0;color:#bfc6c7;background-color:#bfc6c7;width:100%">
<br>
<a href='#top'> Return to top </a>

<p>A Genetic Algorithm (GA)'s goal is to search and optimize. To find possible solutions to a given problem.</p>

<p style="margin-left: 5px"> <b>Route design:</b>
<br>
<br>
<center><img src="https://raw.githubusercontent.com/RohitUttam/Genetic-Algorithms/master/images/Route.png" width="350"/></center>
<br>
<br>
<b>Scheduling planner:</b>
<br>
<br>
<center><img src="https://raw.githubusercontent.com/RohitUttam/Genetic-Algorithms/master/images/ScheduleOpt.png" alt="Drawing" style="width: 600px;"/></center>
<br>
<br>
<b>Feature selection:</b>
<br>
<br>
<center><img src="https://raw.githubusercontent.com/RohitUttam/Genetic-Algorithms/master/images/FeatureSelection.png" alt="Drawing" style="width: 550px;"/>
<br>
[1] Genetic algorithms as a strategy for feature selection. Leardi, Riccardo, R. Boggia, and M. Terrile (1992)}</center>
<br>
<br>
<b>Building graph structures:</b>
<br>
<br>
<br>
<center>Neural network structure learning:
<br>
<img src="https://raw.githubusercontent.com/RohitUttam/Genetic-Algorithms/master/images/NN_structure.png" alt="Drawing" width="550"/>
<br>
Neural network parameter learning:
<br>
<img src="https://raw.githubusercontent.com/RohitUttam/Genetic-Algorithms/master/images/NN_param.png" alt="Drawing" width="550"/>
<br>
[2] Cooperative binary-real coded genetic algorithms for generating and adapting artificial neural networks. Barrios et al. (2003)</center>
<br>
<br>
<br>
<center>Structure Learning of Bayesian Networks (DAG)
<br>
<img src="https://raw.githubusercontent.com/RohitUttam/Genetic-Algorithms/master/images/BN.png" alt="Drawing" width="300""/>
<br>
[3] Structure learning of Bayesian networks by genetic algorithms. Larrañaga, Poza, Yurramendi, Murga and Kuijpers (1996)
</center>
<br>
<br>
<br>
<b>Real life applications:</b>
<br>
<br>
A notable real-life application of Genetic Algorithms (GA) lead to the creation of a very specific anthena by NASA:
<br>
<br>
<center><img src="https://raw.githubusercontent.com/RohitUttam/Genetic-Algorithms/master/images/nasa.png" alt="Drawing" width="500""/>
<br>
[4] Automated antenna design with evolutionary algorithms. Hornby, Gregory, Al Globus, Derek Linden, and Jason Lohn (2006)
</center>
</p>











<br>
<br>
<a id='Intro'></a>
<p style="margin-left: 5px; font-size: 1.4em"> <b>2. Introduction to optimization</b></p>
<hr style="height:1.5px;border-width:0;color:#bfc6c7;background-color:#bfc6c7;width:100%">
<br><a href='#top'> Return to top </a>
<br>
An <b> optimization problem </b> consists of maximizing (or equivalently, minimizing) a function by choosing input values from within an allowed set and computing the value of the function. More generally, optimization focuses on <b>finding "best possible values"</b> of some objective function given a defined domain:
<br>
<br>
<br>
<center>
<img src="https://latex.codecogs.com/gif.latex?\max_{x}f(x)&space;\quad&space;(\textrm{or}\min_{x}f(x))" title="\max_{x}f(x) \quad (\textrm{or}\min_{x}f(x))" />
<br>
<br>
<img src="https://latex.codecogs.com/gif.latex?subject\hspace{0.1cm}to\hspace{0.1cm}g_i(x)\leq0&space;\hspace{0.1cm}&space;\forall{i}" title="subject\hspace{0.1cm}to\hspace{0.1cm}g_i(x)\leq0 \hspace{0.1cm} \forall{i}" />
</center><br>
<br>
<a id='globallocal'></a>
<p style="margin-left: 20px; font-size: 1.4em"> <b>2.1. Global vs local optima</b></p>
<p style="margin-left: 20px"> <a href='#top'> Return to top </a></p>
<center><img src="https://raw.githubusercontent.com/RohitUttam/Genetic-Algorithms/master/images/LocalGlobal.png" alt="Drawing" width="620"/></center>
<br>
<br>
<a id='searchspace'></a>
<p style="margin-left: 20px; font-size: 1.4em"> <b>2.2. Search space</b></p>
<p style="margin-left: 20px"> <a href='#top'> Return to top </a></p>
<p style="margin-left: 20px">A search space within the optimization framework is the domain of the function to be optimized. Search spaces for objective functions are not always smooth, a few examples:
<br>
· Unimodal: This is the optimal search space, where gradients are smooth
<br>
· Needle in a Haystack: It can be easy to miss the peak
<br>
· Noisy: spaces with lots of variations can lead to local optima
<br>
· Deceptive: searching with gradient type algorithm can be deceptive</p>
<br>
<center><img src="https://raw.githubusercontent.com/RohitUttam/Genetic-Algorithms/master/images/Searchspace.png" alt="Drawing" width="500""/></center>
<br>
<p style="margin-left: 20px">In general, Genetic Algorithms (GA) won't be able to solve every problem. No free lunch theorem (NFL): There is no single algorithm that is superior at solving problems, to all other algorithms in general. Evolutionary algorithms are efficient at solving discontinuous, non-differentiable, multimodal problems, with noise and unconventional search spaces. On the contrary, its effectiveness decreases when facing simpler problems for which specific algorithms have been developed in their resolution.<br>
<br>
<br>
<a id='solutionspace'></a>
<p style="margin-left: 20px; font-size: 1.4em"> <b>2.3. Solution space </b></p>
<p style="margin-left: 20px"> <a href='#top'> Return to top </a></p>
<br>
<p style="margin-left: 20px">The solution space, in general, can be different from the search space.  This is very common in Evolutionary computation where we work with a codification of the solution as symbols or programs in the search space.
Therefore, codification and search space setup, will play a key role in the success of the algorithm. If the search space is larger than the solution space, we will find invalid solutions. If the solution space is larger than the search space, it might happen that we don't find the optimal solution.
<br>
<center><img src="https://raw.githubusercontent.com/RohitUttam/Genetic-Algorithms/master/images/Encoding.png" alt="Drawing" width="500"/></center>

<br>
<a id='ambiguityredundancy'></a>
<p style="margin-left: 20px; font-size: 1.4em"> <b>2.4. Ambiguity vs. Redundancy </b></p>
<p style="margin-left: 20px"> <a href='#top'> Return to top </a></p>
<p style="margin-left: 20px"><b>Redundancy:</b><br>
Different encodings can code the same solution. When this happens, there are different symbols for the same solution so it is possible to find it more quickly (has been empirically proven).

<p style="margin-left: 20px"><b>Ambiguity:</b><br>
Different solutions can be encoded by the same symbols.
<br>
<center><img src="https://raw.githubusercontent.com/RohitUttam/Genetic-Algorithms/master/images/Redundancy.png" alt="Drawing" width="200"/></center>

<a id='explorationexplotation'></a>
<p style="margin-left: 20px; font-size: 1.4em"> <b>2.5. Exploration vs Exploitation </b></p>
<p style="margin-left: 20px"> <a href='#top'> Return to top </a></p>
<p style="margin-left: 20px"><b>Exploitation (local search):</b><br>Consists of sampling a limited (but already identified as promising) region of the search space with the hope of improving an existing solution. This operation, then, tries to intensify (refine) the search in the neighbourhood of the existing solution.
<br>
<br>
<img align="center" src="https://raw.githubusercontent.com/RohitUttam/Genetic-Algorithms/master/images/Exploitation.png" alt="Drawing" width="150""/>
<br>
<p style="margin-left: 20px"><b>Exploration (global search):</b><br>Consists of sampling a much larger portion of the search space with the hope of finding other promising solutions that are yet to be refined. This operation, tries to diversify the search in order to avoid local optimums.
<br>
<img align="center" src="https://raw.githubusercontent.com/RohitUttam/Genetic-Algorithms/master/images/Exploration.png" alt="Drawing" width="180"/>
</p>
<br>
<p style="margin-left: 20px">A good search algorithm will have to keep a balance in order to explore a large search space at the beginning and narrow down the search in final iterations.
















<br>
<br>
<br>
<a id='Methods'></a>
<p style="margin-left: 5px; font-size: 1.4em"> <b> 3. Exact vs. approximate methods</b></p>
<hr style="height:1.5px;border-width:0;color:#bfc6c7;background-color:#bfc6c7;width:100%">
<br>
<a href='#top'> Return to top </a><br>
The trade-off criteria for deciding whether to use a heuristic for solving a given problem include the following:
* Objective function: Not required to be fully defined, but we need to be able to evaluate it.
* Optimality: Metaheuristics do not guarantee that a globally optimal solution can be found on some class of problems, but it can provide good approximations in short time.
* Search space: if search space is large or extremely large, classic methods are inefficient.
* Completeness:  Many heuristics are only meant to find one solution.
* Confidence: If a confidence interval is required other approaches can provide it.
* Performance: Some heuristics converge faster than others while some are only marginally quicker than classic methods.

 <br>
They are increasingly used in intermediate and large search space. In optimization complexity terms, they are often used to solve NP Problems and large combinatorial problems in P too.
<br>    
<center><img src="https://raw.githubusercontent.com/RohitUttam/Genetic-Algorithms/master/images/complexity.png" alt="Drawing" style="width: 300px;"/></center>

<br> 
<br> 
<br> 
<center><img src="https://raw.githubusercontent.com/RohitUttam/Genetic-Algorithms/master/images/MathProg.png" alt="Drawing" style="width: 780px;"/></center>

</p>
<br>
<center><img src="https://raw.githubusercontent.com/RohitUttam/Genetic-Algorithms/master/images/Metaheuristic.png" alt="Drawing" style="width: 680px;"/></p></center>









<br>
<br>
<br>
<a id='GeneticAlgorithms'></a>
<p style="margin-left: 5px; font-size: 1.4em"> <b> 4. Genetic Algorithms</b></p>
<hr style="height:1.5px;border-width:0;color:#bfc6c7;background-color:#bfc6c7;width:100%">
<br>
<a href='#top'> Return to top </a>
<br>
A Genetic Algorithm (GA) is a metaheuristic inspired by the process of natural selection that belongs to the larger class of evolutionary algorithms (EA). It is based on neo-Darwinism paradigm: "survival of the fittest", where most of the history of life can be completely justified by physical processes, populations and species. These processes are:
<br><br>
<p style="margin-left: 40px"><b>Reproduction </b><br>
It involves the transfer of the genetic code of each individual to their offspring.
<br><br>
<b>Mutation </b><br> Due to errors in replication during the genetic information transfer process are unavoidable. In addition, mutation is necessary to include diversity in the species.
<br><br>
<b>Competition (Selection) </b><br> It is a consequence of population growth within a finite physical space where there is no space or energy for everyone. Selection is the result of the reproduction of the species in a competitive environment: only the individuals that are more adapated to the environment, survive (i.e. the fittest).
<br><br>
<a id='Basics'></a>
<p style="margin-left: 20px; font-size: 1.4em"> <b>4.1. Basic concepts</b></p>
<p style="margin-left: 20px"> <a href='#top'> Return to top </a></p>
<p style="margin-left: 20px">
<b>Population</b> 
<br>The set of points in the search space or individuals (possible solutions to the problem) with which a GA works at a given moment in the evolution process.
<br>
<center><img src="https://raw.githubusercontent.com/RohitUttam/Genetic-Algorithms/master/images/Population2.png" alt="Drawing" style="width: 400px;"/></center>
<br>
<p style="margin-left: 20px"><b>Chromosomes (individual)</b><br> 
A chromosome is usually identified as an individual in Genetic Algorithms, although in nature, an individual consists of several chromosomes. Individuals and species can be seen as a duality of their genetic code, the genotype, and their way of expressing themselves with respect to the physical world, the phenotype. The genotype offers the possibility of storing the accumulated experience, acquired from historical information.<br>
<br>In summary:</p>
<p style="margin-left: 60px"><b>Genotype </b><br>The set of genes representing the chromosome (Search space).<br><br>
<b>Phenotype</b><br> The actual physical representation of the chromosome (Solution space).
<p style="margin-left: 20px"><b>Genes</b><br> Set of parameters that represent the individuals that make up a population. Each gene represents a position in the chain.
<br>
<center><img src="https://raw.githubusercontent.com/RohitUttam/Genetic-Algorithms/master/images/Individual2.png" alt="Drawing" style="width: 400px;"/></center>
<br>
<p style="margin-left: 20px"><b>Allele</b><br>  Possible values of each gene (symbols). The number of alleles is equal to the cardinality of the alphabet used (m).
<br>
<center><img src="https://raw.githubusercontent.com/RohitUttam/Genetic-Algorithms/master/images/Gene2.png" alt="Drawing" style="width: 180px;"/></center>
<br>
<a id='Lifecycle'></a>
<p style="margin-left: 20px; font-size: 1.4em"> <b>4.2. Evolutionary lifecycle</b></p>
<p style="margin-left: 20px"> <a href='#top'> Return to top </a></p>

<p style="margin-left: 20px">As part of Evolutionary Algorithms, GA use mechanisms inspired by biological evolution, such as reproduction, mutation, recombination, and selection. Candidate solutions to the optimization problem play the role of individuals in a population, and the fitness function determines the quality of the solutions (see also loss function). Evolution of the population then takes place after the repeated application of the above operators.
<br>
<center><video style="display:block; margin: 0 auto;" class='center' src="https://raw.githubusercontent.com/RohitUttam/Genetic-Algorithms/master/videos/Lifecycle2.mov" alt="test" controls loop width="600"></video></center>
<br>
<a id='ProsCons'></a>
<p style="margin-left: 20px; font-size: 1.4em"> <b>4.3. Pros and Cons</b></p>
<p style="margin-left: 20px"> <a href='#top'> Return to top </a></p>
<br>
<center><img src="https://raw.githubusercontent.com/RohitUttam/Genetic-Algorithms/master/images/ProsCons.png" alt="Drawing" style="width: 400px;"/></center>
<br>










<br>
<br>
<br>
<a id='Game'></a>
<p style="margin-left: 5px; font-size: 1.4em"> <b> 5. Example: Target sum game</b></p>
<hr style="height:1.5px;border-width:0;color:#bfc6c7;background-color:#bfc6c7;width:100%">
<br>
<a href='#top'> Return to top </a>
<br>
<center><video style="display:block; margin: 0 auto;" class='center' src="https://raw.githubusercontent.com/RohitUttam/Genetic-Algorithms/master/videos/TargetNumber.mov" alt="test" controls loop width="600"></video></center>
<br>
<br>
<b>Problem statement:</b>
<br>
The goal of the game is to find the right operators (+,-,*,<img src="https://latex.codecogs.com/gif.latex?\div" title="\div" />) to place them between the numbers in order for to achieve target value (or as close as possible).
Note: To keep it simple the operations are completed in sequencial order (there is no BODMAS order)
<br>
<br>
<br>
<center>
<img src="https://latex.codecogs.com/gif.latex?75\hspace{0.3cm}&space;\_&space;\hspace{0.3cm}&space;3\hspace{0.3cm}&space;\_&space;\hspace{0.3cm}&space;1\hspace{0.3cm}&space;\_&space;\hspace{0.3cm}&space;4\hspace{0.3cm}&space;\_&space;\hspace{0.3cm}&space;50\hspace{0.3cm}&space;\_&space;\hspace{0.3cm}&space;6\hspace{0.3cm}&space;\_&space;\hspace{0.3cm}&space;12\hspace{0.3cm}&space;\_&space;\hspace{0.3cm}&space;8\hspace{0.3cm}&space;=&space;\hspace{0.3cm}&space;target" title="75\hspace{0.3cm} \_ \hspace{0.3cm} 3\hspace{0.3cm} \_ \hspace{0.3cm} 1\hspace{0.3cm} \_ \hspace{0.3cm} 4\hspace{0.3cm} \_ \hspace{0.3cm} 50\hspace{0.3cm} \_ \hspace{0.3cm} 6\hspace{0.3cm} \_ \hspace{0.3cm} 12\hspace{0.3cm} \_ \hspace{0.3cm} 8\hspace{0.3cm} = \hspace{0.3cm} target" /></center>
<br>

<a id='Encoding'></a>
<p style="margin-left: 20px; font-size: 1.4em"> <b>5.1. Encoding</b></p>
<p style="margin-left: 20px"> <a href='#top'> Return to top </a></p>

<p style="margin-left: 20px">In order to build genetic chains, an encoding of the solution onto the search space must be provided. A careful choice, will help the algorithm converge to the right solution and yield faster performance.
<br>
<center><img src="https://raw.githubusercontent.com/RohitUttam/Genetic-Algorithms/master/images/Encoding2.png" alt="Drawing" style="width: 300px;"/></center>
<br>
<center><img src="https://raw.githubusercontent.com/RohitUttam/Genetic-Algorithms/master/images/EncodingTarget.png" alt="Drawing" style="width: 350px;"/></center>
<br>
<br>
<a id='Initialization'></a>
<p style="margin-left: 20px; font-size: 1.4em"> <b>5.2. Initialization</b></p>
<p style="margin-left: 20px"> <a href='#top'> Return to top </a></p>
<p style="margin-left: 20px">A population is initialized randomly in the first iteration (generation) of every simulation. The hyperparameter $P$, size of population has to be chosen before execution while $S$, size of individuals is fixed by the paramaters of the problem we are trying to optimize (in this case the numbers):
<br>
<center><img src="https://raw.githubusercontent.com/RohitUttam/Genetic-Algorithms/master/images/Initialization.png" alt="Drawing" style="width: 500px;"/></center>
<br>

<a id='Fitness'></a>
<p style="margin-left: 20px; font-size: 1.4em"> <b>5.3. Fitness</b></p>
<p style="margin-left: 20px"> <a href='#top'> Return to top </a></p>

<p style="margin-left: 20px"> At every timestep (generation), the population has to be evaluated through a $Fitness$ function. This $Fitness$ function will provide a score for each individual, indicating if the individual is a good candidate for the overall solution or not. Keep in mind that this function will me evaluated $PxG$ times, where $P$ is the size of population and $G$, the number of generations. So a Fitness function has to be fast and efficient.
<br>
<center><img src="https://raw.githubusercontent.com/RohitUttam/Genetic-Algorithms/master/images/Fitness.png" alt="Drawing" style="width: 500px;"/></center>
<br>
<a id='Selection'></a>
<p style="margin-left: 20px; font-size: 1.4em"> <b>5.4. Selection</b></p>
<p style="margin-left: 20px"> <a href='#top'> Return to top </a></p>

<p style="margin-left: 20px">Some individuals are selected to create new generations of individuals.
The idea is for individuals with greater level of adaptation (i.e. fitness) to be chosen with higher probability, and therefore passing on genetic information to their offsprings.
</p>

<center><img src="https://raw.githubusercontent.com/RohitUttam/Genetic-Algorithms/master/images/Selection.png" alt="Drawing" style="width: 400px;"/></center>

<a id='Crossover'></a>
<p style="margin-left: 20px; font-size: 1.4em"> <b>5.5. Crossover</b></p>
<p style="margin-left: 20px"> <a href='#top'> Return to top </a></p>
<p style="margin-left: 20px"> Once the mating individuals are chosen, it remains to define how they will reproduce to develop new individuals. This is the most valuable operator in the Genetic Algorithm's architecture. It will determine the success of the search problem.</p>
<br>
<a id='SinglePoint'></a>
<p style="margin-left: 40px"> <b>5.5.1. Single Point Crossover</b></p>
<p style="margin-left: 40px"> <a href='#top'> Return to top </a></p>

<p style="margin-left: 40px"> New generations of individuals are generated from the selected individuals,by choosing one crossover point at random. The goal is to preserve large building blocks from fittest individuals in current generation.

<br>
<br>
<center><img src="https://latex.codecogs.com/gif.latex?\text{Single&space;Point&space;Crossover}" title="\text{Single Point Crossover}" />
<br><img src="https://raw.githubusercontent.com/RohitUttam/Genetic-Algorithms/master/images/SinglePoint.png" alt="Drawing" style="width: 400px;"/></p></center>

<a id='Uniform'></a>
<p style="margin-left: 40px"> <b>5.4.2. Uniform crossover</b></p>
<p style="margin-left: 40px"> <a href='#top'> Return to top </a></p>
<p style="margin-left: 40px">It has been proven in the literature that adding more crossover points helps the algorithms uptil a certain point $\text{(DeJong, 75 )}$. In general, more than two crossover points reduces performance. The reason for this is that the building blocks will of course be disrupted more often, however there is a tradeoff, it will help adding diversity and exploring further the search space.
<br>
<center><img src="https://raw.githubusercontent.com/RohitUttam/Genetic-Algorithms/master/images/UniformCrossover.png" alt="Drawing" style="width: 500px;" class="center"></p></center>

<a id='Mutation'></a>
<p style="margin-left: 20px; font-size: 1.4em"> <b>5.6. Mutation</b></p>
<p style="margin-left: 20px"> <a href='#top'> Return to top </a></p>
<p style="margin-left: 20px">The mutation operator modifies the gene of the individuals stochastically. The goal of this operator is to increase the search space in which the algorithm is looking for the solution. 
Use of mutation operators are usually kept below p=0.05, as they introduce randomness to the algorithm.

<center><img src="https://raw.githubusercontent.com/RohitUttam/Genetic-Algorithms/master/images/Mutation.png" alt="Drawing" style="width: 500px;"/></p></center>
<a id='Replacement'></a>
<p style="margin-left: 20px; font-size: 1.4em"> <b>5.7. Replacement</b></p>
<p style="margin-left: 20px"> <a href='#top'> Return to top </a>
<br>
Once new individuals have been generated, it remains to see, which ones are kept in the population. In general, the population size is invariant with time. Therefore, some individuals, will have to be replaced in order to keep the fittest individuals with higher probability.
<br>
<br>

</p>
<a id='SinglePoint'></a>
<p style="margin-left: 40px"> <b>5.7.1. Elitist Replacement</b></p>
<p style="margin-left: 40px"> <a href='#top'> Return to top </a></p>
<p style="margin-left: 40px"> An elitist replacement operator is completely deterministic. It chooses only the individuals with the highest fitness among previous and current generation of individuals, to create next iteration’s population.
The rest are eliminated.
<br>
<center><img src="https://raw.githubusercontent.com/RohitUttam/Genetic-Algorithms/master/images/Elitist.png" alt="Drawing" style="width: 650px;" class="center"></center>
</p>
<a id='SinglePoint'></a>
<p style="margin-left: 40px"> <b>5.7.2. Simple Reduction</b></p>
<p style="margin-left: 40px"> <a href='#top'> Return to top </a></p>
<p style="margin-left: 40px"> A simple reduction replacement operator is stochastic. It chooses individuals with the lower fitness to be replaced with higher probability, among previous and current generation of individuals.
<br>
<center><img src="https://raw.githubusercontent.com/RohitUttam/Genetic-Algorithms/master/images/Simplereduction.png" alt="Drawing" style="width: 650px;"/>
</center>
</p>
<br>
<center><img src="https://raw.githubusercontent.com/RohitUttam/Genetic-Algorithms/master/images/replacement.png" alt="Drawing" style="width: 450px;"/></center>

</p>

<br>
<br>
<br>
<a id='Convergence'></a>
<p style="margin-left: 20px; font-size: 1.4em"> <b>EXTRA: Convergence</b></p>
<hr style="height:1.5px;border-width:0;color:#bfc6c7;background-color:#bfc6c7;width:100%"></hr>

<p style="margin-left: 20px"><a href='#top'> Return to top </a></p>

<p style="margin-left: 20px">Regarding convergence, there are various metrics that study convergence of Genetic Algorithms [5]. Two common metrics  are:

<center>
<img src="https://latex.codecogs.com/gif.latex?\textrm{Offline&space;measure}\hspace{0.5cm}&space;m^{}(T)=\dfrac{1}{T}\sum_{t=1}^{T}(f(I^{}(t))" title="\textrm{Offline measure}\hspace{0.5cm} m^{}(T)=\dfrac{1}{T}\sum_{t=1}^{T}(f(I^{}(t))" />
</center>

<p style="margin-left: 20px">Where <img src="https://latex.codecogs.com/gif.latex?f(I^{*}(t))" title="f(I^{*}(t))" /> is the fitness of the fittest individual at generation t.
Therefore, offline measurement can be interpreted as a progress indicator. After a number of generations, this measurement can be near the optima or not, however the velocity of when this happens indicates the ability of the algorithm to establish itself near the solution. It can serve the purpose of being used as a stopping criteria (i.e. when difference between generations is close).

<center>
<img src="https://latex.codecogs.com/gif.latex?\textrm{Online&space;measure}\hspace{0.7cm}&space;m(T)=\dfrac{1}{T}\sum_{t=1}^{T}F(t)" title="\textrm{Online measure}\hspace{0.7cm} m(T)=\dfrac{1}{T}\sum_{t=1}^{T}F(t)" /></center>

<p style="margin-left: 20px">Online measurement is defined as the mean, for the objective function, till evaluation. Where F(t) is the mean of fitness of all individuals available at generation t.

<br>
<br>
<a id='Biblio'></a>
<p style="margin-left: 20px; font-size: 1.4em"> <b>Bibliography</b></p>
<hr style="height:1.5px;border-width:0;color:#bfc6c7;background-color:#bfc6c7;width:100%">
<p style="margin-left: 20px">
<a href='#top'> Return to top </a>
<br>

[1] Leardi, Riccardo, R. Boggia, and M. Terrile. Genetic algorithms as a strategy for feature selection. Journal of chemometrics 6, no. 5 (1992): 267-281.
<br>
<br>
[2] Barrios, Dolores, Alberto Carrascal, Daniel Manrique, and Juan Rios. Cooperative binary-real coded genetic algorithms for generating and adapting artificial neural networks. Neural Computing & Applications 12, no. 2 (2003): 49-60.
<br>
<br>
[3] Larranaga, Pedro, Mikel Poza, Yosu Yurramendi, Roberto H. Murga, and Cindy M. H. Kuijpers. Structure learning of Bayesian networks by genetic algorithms: A performance analysis of control parameters. IEEE transactions on pattern analysis and machine intelligence 18, no. 9 (1996): 912-926.
<br>
<br>
[4]  Hornby, Gregory, Al Globus, Derek Linden, and Jason Lohn. Automated antenna design with evolutionary algorithms. In Space 2006, p. 7242. 2006.
<br>
<br>
[5] DeJong, K.The Analysis and behaviour of a Class of Genetic Adapative Systems. PhD Thesis, University of Michigan, 1975.
</p>
<!-- Author: Rohit Uttamchandani -->
<!-- Title: Introduction to Genetic Algorithms -->
<!-- Date: 28.05.2021 -->

</dl>