/* *****************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
package nyway.nd4j.samples.vgg16.flower

import nyway.nd4j.samples.Samples
import org.deeplearning4j.nn.graph.ComputationGraph
import org.nd4j.evaluation.classification.Evaluation
import java.io.File
import java.io.IOException

object LoadVgg16Flower {

    @Throws(IOException::class)
    @JvmStatic
    fun main(args: Array<String>) {
        val vgg16OnFlower = ComputationGraph.load(File("${Samples.modelFolder}/vgg16/Vgg16OnFlower.zip"), true)
        FlowerDataSetIterator.setup(15, EditLastLayerOthersFrozen.trainPerc)
        val testIter = FlowerDataSetIterator.testIterator()
        var eval: Evaluation
        eval = vgg16OnFlower.evaluate(testIter)
        println(eval.stats())
    }
}