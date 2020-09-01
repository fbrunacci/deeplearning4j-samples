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
package nyway.nd4j.samples.recurrent

import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import java.io.File
import java.io.IOException
import java.nio.charset.Charset
import java.nio.file.Files
import java.util.*

/** A simple DataSetIterator for use in the LSTMCharModellingExample.
 * Given a text file and a few options, generate feature vectors and labels for training,
 * where we want to predict the next character in the sequence.<br></br>
 * This is done by randomly choosing a position in the text file, at offsets of 0, exampleLength, 2*exampleLength, etc
 * to start each sequence. Then we convert each character to an index, i.e., a one-hot vector.
 * Then the character 'a' becomes [1,0,0,0,...], 'b' becomes [0,1,0,0,...], etc
 *
 * Feature vectors and labels are both one-hot vectors of same length
 * @author Alex Black
 */
class CharacterIterator(textFilePath: String, textFileEncoding: Charset?, miniBatchSize: Int, exampleLength: Int,
                        validCharacters: CharArray, rng: Random, commentChars: String?) : DataSetIterator {
    //Valid characters
    private val validCharacters: CharArray

    //Maps each character to an index ind the input/output
    private val charToIdxMap: MutableMap<Char, Int>

    //All characters of the input file (after filtering to only those that are valid
    private val fileCharacters: CharArray

    //Length of each example/minibatch (number of characters)
    private val exampleLength: Int

    //Size of each minibatch (number of examples)
    private val miniBatchSize: Int
    private val rng: Random

    //Offsets for the start of each example
    private val exampleStartOffsets = LinkedList<Int>()

    /**
     * @param textFilePath Path to text file to use for generating samples
     * @param textFileEncoding Encoding of the text file. Can try Charset.defaultCharset()
     * @param miniBatchSize Number of examples per mini-batch
     * @param exampleLength Number of characters in each input/output vector
     * @param validCharacters Character array of valid characters. Characters not present in this array will be removed
     * @param rng Random number generator, for repeatability if required
     * @throws IOException If text file cannot  be loaded
     */
    internal constructor(textFilePath: String, textFileEncoding: Charset?, miniBatchSize: Int, exampleLength: Int,
                         validCharacters: CharArray, rng: Random) : this(textFilePath, textFileEncoding, miniBatchSize, exampleLength, validCharacters, rng, null) {
    }

    fun convertIndexToCharacter(idx: Int): Char {
        return validCharacters[idx]
    }

    fun convertCharacterToIndex(c: Char): Int {
        return charToIdxMap[c]!!
    }

    val randomCharacter: Char
        get() = validCharacters[(rng.nextDouble() * validCharacters.size).toInt()]

    override fun hasNext(): Boolean {
        return exampleStartOffsets.size > 0
    }

    override fun next(): DataSet {
        return next(miniBatchSize)
    }

    override fun next(num: Int): DataSet {
        if (exampleStartOffsets.size == 0) throw NoSuchElementException()
        val currMinibatchSize = Math.min(num, exampleStartOffsets.size)
        //Allocate space:
        //Note the order here:
        // dimension 0 = number of examples in minibatch
        // dimension 1 = size of each vector (i.e., number of characters)
        // dimension 2 = length of each time series/example
        //Why 'f' order here? See http://deeplearning4j.org/usingrnns.html#data section "Alternative: Implementing a custom DataSetIterator"
        val input = Nd4j.create(intArrayOf(currMinibatchSize, validCharacters.size, exampleLength), 'f')
        val labels = Nd4j.create(intArrayOf(currMinibatchSize, validCharacters.size, exampleLength), 'f')
        for (i in 0 until currMinibatchSize) {
            val startIdx = exampleStartOffsets.removeFirst()
            val endIdx = startIdx + exampleLength
            var currCharIdx = charToIdxMap[fileCharacters[startIdx]]!! //Current input
            var c = 0
            var j = startIdx + 1
            while (j < endIdx) {
                val nextCharIdx = charToIdxMap[fileCharacters[j]]!! //Next character to predict
                input.putScalar(intArrayOf(i, currCharIdx, c), 1.0)
                labels.putScalar(intArrayOf(i, nextCharIdx, c), 1.0)
                currCharIdx = nextCharIdx
                j++
                c++
            }
        }
        return DataSet(input, labels)
    }

    private fun totalExamples(): Int {
        return (fileCharacters.size - 1) / miniBatchSize - 2
    }

    override fun inputColumns(): Int {
        return validCharacters.size
    }

    override fun totalOutcomes(): Int {
        return validCharacters.size
    }

    override fun reset() {
        exampleStartOffsets.clear()
        initializeOffsets()
    }

    private fun initializeOffsets() {
        //This defines the order in which parts of the file are fetched
        val nMinibatchesPerEpoch = (fileCharacters.size - 1) / exampleLength - 2 //-2: for end index, and for partial example
        for (i in 0 until nMinibatchesPerEpoch) {
            exampleStartOffsets.add(i * exampleLength)
        }
        Collections.shuffle(exampleStartOffsets, rng)
    }

    override fun resetSupported(): Boolean {
        return true
    }

    override fun asyncSupported(): Boolean {
        return true
    }

    override fun batch(): Int {
        return miniBatchSize
    }

    fun cursor(): Int {
        return totalExamples() - exampleStartOffsets.size
    }

    fun numExamples(): Int {
        return totalExamples()
    }

    override fun setPreProcessor(preProcessor: DataSetPreProcessor) {
        throw UnsupportedOperationException("Not implemented")
    }

    override fun getPreProcessor(): DataSetPreProcessor {
        throw UnsupportedOperationException("Not implemented")
    }

    override fun getLabels(): List<String> {
        throw UnsupportedOperationException("Not implemented")
    }

    override fun remove() {
        throw UnsupportedOperationException()
    }

    companion object {
        /** A minimal character set, with a-z, A-Z, 0-9 and common punctuation etc  */
        val minimalCharacterSet: CharArray
            get() {
                val validChars: MutableList<Char> = LinkedList()
                run {
                    var c = 'a'
                    while (c <= 'z') {
                        validChars.add(c)
                        c++
                    }
                }
                run {
                    var c = 'A'
                    while (c <= 'Z') {
                        validChars.add(c)
                        c++
                    }
                }
                var c = '0'
                while (c <= '9') {
                    validChars.add(c)
                    c++
                }
                val temp = charArrayOf('!', '&', '(', ')', '?', '-', '\'', '"', ',', '.', ':', ';', ' ', '\n', '\t')
                for (c in temp) validChars.add(c)
                val out = CharArray(validChars.size)
                var i = 0
                for (c in validChars) out[i++] = c
                return out
            }

        /** As per getMinimalCharacterSet(), but with a few extra characters  */
        val defaultCharacterSet: CharArray
            get() {
                val validChars: MutableList<Char> = LinkedList()
                for (c in minimalCharacterSet) validChars.add(c)
                val additionalChars = charArrayOf('@', '#', '$', '%', '^', '*', '{', '}', '[', ']', '/', '+', '_',
                        '\\', '|', '<', '>')
                for (c in additionalChars) validChars.add(c)
                val out = CharArray(validChars.size)
                var i = 0
                for (c in validChars) out[i++] = c
                return out
            }
    }

    /**
     * @param textFilePath Path to text file to use for generating samples
     * @param textFileEncoding Encoding of the text file. Can try Charset.defaultCharset()
     * @param miniBatchSize Number of examples per mini-batch
     * @param exampleLength Number of characters in each input/output vector
     * @param validCharacters Character array of valid characters. Characters not present in this array will be removed
     * @param rng Random number generator, for repeatability if required
     * @param commentChars if non-null, lines starting with this string are skipped.
     * @throws IOException If text file cannot  be loaded
     */
    init {
        if (!File(textFilePath).exists()) throw IOException("Could not access file (does not exist): $textFilePath")
        require(miniBatchSize > 0) { "Invalid miniBatchSize (must be >0)" }
        this.validCharacters = validCharacters
        this.exampleLength = exampleLength
        this.miniBatchSize = miniBatchSize
        this.rng = rng

        //Store valid characters is a map for later use in vectorization
        charToIdxMap = HashMap()
        for (i in validCharacters.indices) charToIdxMap[validCharacters[i]] = i

        //Load file and convert contents to a char[]
        val newLineValid = charToIdxMap.containsKey('\n')
        var lines = Files.readAllLines(File(textFilePath).toPath(), textFileEncoding)
        if (commentChars != null) {
            val withoutComments: MutableList<String> = ArrayList()
            for (line in lines) {
                if (!line.startsWith(commentChars)) {
                    withoutComments.add(line)
                }
            }
            lines = withoutComments
        }
        var maxSize = lines.size //add lines.size() to account for newline characters at end of each line
        for (s in lines) maxSize += s.length
        val characters = CharArray(maxSize)
        var currIdx = 0
        for (s in lines) {
            val thisLine = s.toCharArray()
            for (aThisLine in thisLine) {
                if (!charToIdxMap.containsKey(aThisLine)) continue
                characters[currIdx++] = aThisLine
            }
            if (newLineValid) characters[currIdx++] = '\n'
        }
        fileCharacters = if (currIdx == characters.size) {
            characters
        } else {
            Arrays.copyOfRange(characters, 0, currIdx)
        }
        require(exampleLength < fileCharacters.size) {
            ("exampleLength=" + exampleLength
                    + " cannot exceed number of valid characters in file (" + fileCharacters.size + ")")
        }
        val nRemoved = maxSize - fileCharacters.size
        println("Loaded and converted file: " + fileCharacters.size + " valid characters of "
                + maxSize + " total characters (" + nRemoved + " removed)")
        initializeOffsets()
    }
}