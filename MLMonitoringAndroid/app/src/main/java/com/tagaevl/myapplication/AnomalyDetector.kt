package com.tagaevl.myapplication

import android.content.Context
import android.content.res.AssetManager
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import org.tensorflow.lite.flex.FlexDelegate

class AnomalyDetector(context: Context) {

    private var interpreter: Interpreter

    init {
       // val modelFile = loadModelFile(context.assets,"anomaly_detector.tflite") //context.assets.open("anomaly_detector.tflite")
        val modelFile = context.assets.open("anomaly_detector.tflite")
        val modelData = modelFile.readBytes()
        val byteBuffer = ByteBuffer.allocateDirect(modelData.size).order(ByteOrder.nativeOrder())
        byteBuffer.put(modelData)
        val options = Interpreter.Options()

        options.addDelegate(FlexDelegate())  // Enables SELECT_TF_OPS
        interpreter = Interpreter(byteBuffer)
    }

    fun predict(sequence: FloatArray): Float {
        val input = arrayOf(sequence.map { arrayOf(it) }.toTypedArray())
        val output = Array(1) { FloatArray(1) }
        interpreter.run(input, output)
        return output[0][0]  // The predicted value
    }

    @Throws(IOException::class)
    private fun loadModelFile(assetManager: AssetManager, filename: String): ByteBuffer {
        val fileDescriptor = assetManager.openFd(filename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
}
