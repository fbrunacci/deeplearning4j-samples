package nyway.utils

import java.util.*

object YesNo {
    @JvmStatic
    fun main(args: Array<String>) {
        if (askYesNo("Do you wish to proceed? (Y/N): ")) {
            println("Okay...")
        }
        if (askYesNo("Delete all Files? (Okay/Cancel): ", "okay", "cancel")) {
            println("Make it so")
        }
    }

    @JvmOverloads
    fun askYesNo(question: String?, positive: String = "Y", negative: String = "N"): Boolean {
        val input = Scanner(System.`in`)
        var answer: String
        do {
            print(question)
            answer = input.next().trim { it <= ' ' }
        } while ( !(answer.equals(positive, true) || answer.equals(negative, true)) )
        // Assess if we match a positive response
        return answer.equals(positive, true)
    }
}