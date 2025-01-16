package com.tagaevl.myapplication

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import com.tagaevl.myapplication.Utils.input
import com.tagaevl.myapplication.Utils.inputNUM
import com.tagaevl.myapplication.Utils.inputNUM2
import com.tagaevl.myapplication.Utils.normal
import com.tagaevl.myapplication.ui.theme.MLMonitoringAndroidTheme
import kotlin.random.Random

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        // Example usage
        val detector = AnomalyDetector2(applicationContext)

        // Example sequence (ensure the length matches the input sequence length during training)
        var testSequence = arrayListOf(0.1f, 0.2f, 0.3f) // Replace with real data
        repeat(500) {
            testSequence.add((-1.. 1).random().toFloat())
        }
        //testSequence = (testSequence + arrayListOf(0, 1, 100)) as ArrayList<Float>

        //val input = "[ 0.49671415 -0.1382643   0.64768854  1.52302986 -0.23415337 -0.23413696 1.57921282  0.76743473 -0.46947439  0.54256004 -0.46341769 -0.46572975 0.24196227 -1.91328024 -1.72491783 -0.56228753 -1.01283112  0.31424733 -0.90802408 -1.4123037   1.46564877 -0.2257763   0.0675282  -1.42474819 -0.54438272  0.11092259 -1.15099358  0.37569802 -0.60063869 -0.29169375 -0.60170661  1.85227818 -0.01349722 -1.05771093  0.82254491 -1.22084365 0.2088636  -1.95967012 -1.32818605  0.19686124  0.73846658  0.17136828 -0.11564828 -0.3011037  -1.47852199 -0.71984421 -0.46063877  1.05712223 0.34361829 -1.76304016  0.32408397 -0.38508228 -0.676922    0.61167629 1.03099952  0.93128012 -0.83921752 -0.30921238  0.33126343  0.97554513 -0.47917424 -0.18565898 -1.10633497 -1.19620662  0.81252582  1.35624003 -0.07201012  1.0035329   0.36163603 -0.64511975  0.36139561  1.53803657 -0.03582604  1.56464366 -2.6197451   0.8219025   0.08704707 -0.29900735 0.09176078 -1.98756891 -0.21967189  0.35711257  1.47789404 -0.51827022 -0.8084936  -0.50175704  0.91540212  0.32875111 -0.5297602   0.51326743 0.09707755  0.96864499 -0.70205309 -0.32766215 -0.39210815 -1.46351495 0.29612028  0.26105527  0.00511346 -0.23458713]"

        val anomaly = inputNUM //parseToFloatArray(input)

        val prediction = detector.predict(normal + anomaly)
        println(">>>>>>>>>>> ${prediction}")


        if (prediction > 0.5) {  // Define a suitable threshold
            println("Anomaly detected!")
        } else {
            println("Normal pattern.")
        }


        setContent {
            MLMonitoringAndroidTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Greeting(
                        name = "Android",
                        modifier = Modifier.padding(innerPadding)
                    )
                }
            }
        }
    }
}

@Composable
fun Greeting(name: String, modifier: Modifier = Modifier) {
    Text(
        text = "Hello $name!",
        modifier = modifier
    )
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    MLMonitoringAndroidTheme {
        Greeting("Android")
    }
}