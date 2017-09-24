package assignment2

/**
  * Created by Tom Lous on 27/08/2017.
  *
  * @param name          Label for testCase
  * @param epochs        Number of epochs to run.
  * @param learning_rate Learning rate; default = 0.1.
  * @param momentum      Momentum; default = 0.9.
  * @param numhid1       Dimensionality of embedding space; default = 50.
  * @param numhid2       Number of units in hidden layer; default = 200.
  * @param init_wt       Standard deviation of the normal distribution which is sampled to get the initial weights; default = 0.01
  */
case class TrainingCase(
                         name: String,
                         epochs: Int,
                         learning_rate: Double = 0.1,
                         momentum: Double = 0.9,
                         numhid1: Int = 50,
                         numhid2: Int = 200,
                         init_wt: Double = 0.01
                       ){
  override def toString() = name

  val fileName:String = List(name.replaceAll("""\W+""","").toLowerCase,epochs,learning_rate,momentum,numhid1,numhid2,init_wt).mkString("_") + ".log"
}
