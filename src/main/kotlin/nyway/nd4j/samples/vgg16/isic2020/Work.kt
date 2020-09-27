package nyway.nd4j.samples.vgg16.isic2020

import krangl.*
import nyway.nd4j.samples.Samples
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.transform.FlipImageTransform
import org.nd4j.linalg.api.ndarray.INDArray
import java.io.File
import javax.xml.crypto.Data

object Work {

    @JvmStatic
    fun main(args: Array<String>) {

//        val isicFolder = "${Samples.dataFolder}/ISIC_2019"
//
//        val isicDataSet = nyway.nd4j.samples.vgg16.isic2019.ISICDataSet(
//                "$isicFolder/ISIC_2019_Training_Input",
//                "$isicFolder/ISIC_2019_Training_GroundTruth.csv",
//                trainSize = 0.9f)
//
//        val next = isicDataSet.trainIterator().next()
//        println(next.labels)
//

        val df : DataFrame = DataFrame.readCSV("/home/fabien/.deeplearning4j/data/ISIC_2020/Deotte/512/train.csv").shuffle()
        println(df)
        val malignant = df.filter { it["benign_malignant"] eq "malignant" }
        val benign = df.filter { it["benign_malignant"] eq "benign" }
        df.addColumn("folder"){"/home/fabien/.deeplearning4j/data/ISIC_2020/Deotte/512/train/"}


        File("/home/arjun/tutorials/").walk().forEach {
            println(it)
        }

        val benignSized =  benign.shuffle().take(malignant.rows.count()*3)
        println(benignSized)

        val dfx = bindRows(
                malignant.addColumn("flip"){""},
                malignant.addColumn("flip"){"-1"},
                malignant.addColumn("flip"){"1"},
                malignant.addColumn("flip"){"0"},
                benignSized.addColumn("flip"){""}
        )
        println(dfx.select("image_name","flip"))

        fun readCSVTo(fileOrUrl: String): DataFrame {
            return null
        }

    }

    private fun rowToImage(nextRow: DataFrameRow, imageDir: String): INDArray {
        val imageLoader = NativeImageLoader()
        var asWritable = imageLoader.asWritable("$imageDir/${nextRow["image_name"]}.jpg")
        val transformer = when( nextRow["flip"] ) {
            "0" -> FlipImageTransform(0)
            "-1" -> FlipImageTransform(-1)
            "1" -> FlipImageTransform(1)
            else -> null
        }
        asWritable = transformer?.transform(asWritable)
        return imageLoader.asMatrix(asWritable)
    }

}