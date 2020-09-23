package nyway.nd4j.samples

import com.sksamuel.hoplite.ConfigLoader
import java.io.File

data class Folder(val data : String, val model: String)
data class Config(val work_folder: String, val folder: Folder)

object Samples {

    val config = ConfigLoader().loadConfigOrThrow<Config>("/application.yaml")
    var dataFolder = config.folder.data
    var modelFolder = config.folder.model

    @JvmStatic
    fun main(args: Array<String>) {
        println(config)
    }


}