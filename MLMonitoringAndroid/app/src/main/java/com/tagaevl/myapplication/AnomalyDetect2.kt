package com.tagaevl.myapplication

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

class AnomalyDetector2(context: Context) {
    private val interpreter: Interpreter

    init {
        val assetFileDescriptor = context.assets.openFd("anomaly_detector1.tflite")
        val fileInputStream = assetFileDescriptor.createInputStream()
        val modelBuffer = ByteBuffer.allocateDirect(assetFileDescriptor.length.toInt())
        modelBuffer.order(ByteOrder.nativeOrder())
        modelBuffer.put(fileInputStream.readBytes())
        modelBuffer.flip()

        val options = Interpreter.Options()
        interpreter = Interpreter(modelBuffer, options)
    }

    fun predict(sequence: FloatArray): Float {
        // Prepare input tensor [1, sequenceLength, 1]
        val sequenceLength = sequence.size
        val inputTensor = Array(1) { Array(sequenceLength) { FloatArray(1) } }
        for (i in sequence.indices) {
            inputTensor[0][i][0] = sequence[i]
        }

        // Prepare output tensor [1, 1]
        val outputTensor = Array(1) { FloatArray(1) }

        // Run inference
        interpreter.run(inputTensor, outputTensor)

        // Return prediction result
        return outputTensor[0][0]
    }
}
