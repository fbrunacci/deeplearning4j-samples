package nyway.nd4j.samples.listeners

import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.optimize.api.BaseTrainingListener
import java.io.Serializable

class PrintScoreIterationListener(var printIterations : Int = 1, var newLineIterations : Int = 50) : BaseTrainingListener(), Serializable {

    override fun iterationDone(model: Model, iteration: Int, epoch: Int) {
        if (printIterations <= 0) printIterations = 1
        if (iteration % printIterations == 0) {
            val score = model.score()
            print("\rScore at iteration $iteration is $score")
        }
        if (iteration % newLineIterations == 0) {
            println("")
        }
    }

    override fun toString(): String {
        return "PrintScoreIterationListener($printIterations)"
    }
}