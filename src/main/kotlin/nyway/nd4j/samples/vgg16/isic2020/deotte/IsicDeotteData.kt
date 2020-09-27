package nyway.nd4j.samples.vgg16.isic2020.deotte

import krangl.*
import java.io.File

data class IsicDeotteData(val image_name: String,
                          val benign_malignant: String,
                          val anatom_site_general_challenge: String,
                          val sex: String,
                          val age_approx: String,
                          val target: Int) {


    companion object Factory {
        val columns = listOf("image_name", "anatom_site_general_challenge", "sex", "age_approx", "benign_malignant", "target")

        private fun readCSVToIrisDataFrame(fileOrUrl: String): DataFrame {
            return DataFrame.readCSV(fileOrUrl).select(columns).rows.map { row ->
                IsicDeotteData(
                        row["image_name"] as String,
                        row["benign_malignant"] as String,
                        row["anatom_site_general_challenge"] as String,
                        row["sex"] as String,
                        row["age_approx"].toString(),
                        row["target"] as Int)
            }.asDataFrame()
        }

        fun deotteIsisData(workDir: String): DataFrame {
            val df2019: DataFrame = readCSVToIrisDataFrame("$workDir/cdeotte-isic2019-v4-512x512/train.csv")
            val df2020: DataFrame = readCSVToIrisDataFrame("$workDir/cdeotte-isic2020-v3-512x512/train.csv")
            val dfMalignant1: DataFrame = readCSVToIrisDataFrame("$workDir/malignant-v2-512x512/train_malig_1.csv")
            val dfMalignant2: DataFrame = readCSVToIrisDataFrame("$workDir/malignant-v2-512x512/train_malig_2.csv")
            val dfMalignant3: DataFrame = readCSVToIrisDataFrame("$workDir/malignant-v2-512x512/train_malig_3.csv")
            val dfMalignant = bindRows(
                    dfMalignant1,
                    dfMalignant2,
                    dfMalignant3
            )
            val allDataFrame = bindRows(
                    df2019.addColumn("from") { "2019" }.addColumn("jpegDir") { "$workDir/cdeotte-isic2019-v4-512x512/train" },
                    df2020.addColumn("from") { "2020" }.addColumn("jpegDir") { "$workDir/cdeotte-isic2020-v3-512x512/train" },
                    dfMalignant.addColumn("from") { "malignant" }.addColumn("jpegDir") { "$workDir/malignant-v2-512x512/jpeg512" }
            )

            val filteredOnImageExist = allDataFrame.filterByRow { File("${it["jpegDir"]}/${it["image_name"]}.jpg").exists() }
            return filteredOnImageExist.shuffle()
        }
    }
}