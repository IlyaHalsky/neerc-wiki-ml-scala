package lstm

import java.io.{File, IOException}
import java.net.URL
import java.nio.charset.Charset
import java.util.Random

import org.apache.commons.io.FileUtils
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.nn.conf.{BackpropType, MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

import scala.collection.mutable


/** Пример LSTM для генерации символов
  *         Пример: Обучение LSTM RNN для генерации текста по одному символу.
  *         Пример взят из поста
  *         "The Unreasonable Effectiveness of Recurrent Neural Networks"
  *         http://karpathy.github.io/2015/05/21/rnn-effectiveness/
  *         *
  *         Больше информации по RNN в DL4J:
  *         http://deeplearning4j.org/usingrnns
  *         http://deeplearning4j.org/lstm
  *         http://deeplearning4j.org/recurrentnetwork
  */
object GravesLSTMCharModellingExample {
  @throws[Exception]
  def main(args: Array[String]) {
    val lstmLayerSize = 200                     //Размер каждого слоя
    val miniBatchSize = 32                      //Размер батча
    val exampleLength = 1000                    //Длинна примера для обучения, можно увеличивать
    val tbpttLength = 50                        //Длинна обрезанного обратного прохода во времени т.е. параметры обновляются каждые 50 символов
    val numEpochs = 1                           //Количество эпоъ
    val generateSamplesEveryNMinibatches = 10   //Как часто генерировать тестовые разультаты?
    val nSamplesToGenerate = 4                  //Количество тестовых результатов после каждой эпохи
    val nCharactersToSample = 300               //Длинна тестового результата
    val rng = new Random(12345)

    //Подготовка датасета для обучения
    val iter = getShakespeareIterator(miniBatchSize, exampleLength)
    val nOut = iter.totalOutcomes

    //Настройка нейронной сети:
    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(1)
      .learningRate(0.1)
      .rmsDecay(0.95)
      .seed(12345)
      .regularization(true)
      .l2(0.001)
      .weightInit(WeightInit.XAVIER)
      .updater(Updater.RMSPROP)
      .list
      .layer(0, new GravesLSTM.Builder().nIn(iter.inputColumns).nOut(lstmLayerSize)
        .activation(Activation.TANH).build())
      .layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
        .activation(Activation.TANH).build())
      .layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT)
        .activation(Activation.SOFTMAX)  //MCXENT + softmax for classification
        .nIn(lstmLayerSize).nOut(nOut).build())
      .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
      .pretrain(false).backprop(true)
      .build()

    val net = new MultiLayerNetwork(conf)
    net.init()
    net.setListeners(new ScoreIterationListener(1))

    val layers = net.getLayers
    var totalNumParams = 0
    for (i <- layers.indices) {
      val nParams = layers(i).numParams
      println("Number of parameters in layer " + i + ": " + nParams)
      totalNumParams += nParams
    }
    println("Total number of network parameters: " + totalNumParams)

    //Начать обучени и генерировать тестовые примеры
    var miniBatchNumber = 0
    var i = 0
    for (i <- 0 until numEpochs) {
      while (iter.hasNext) {
        val ds = iter.next
        net.fit(ds)
        if (miniBatchNumber % generateSamplesEveryNMinibatches == 0) {
          println("--------------------")
          println("Completed " + miniBatchNumber + " minibatches of size " + miniBatchSize + "x" + exampleLength + " characters")
          println("Sampling characters from network given initialization \"" +
            (if (Option(generationInitialization).isEmpty) "" else generationInitialization) + "\"")
          val samples = sampleCharactersFromNetwork(generationInitialization, net, iter, rng, nCharactersToSample, nSamplesToGenerate)
          for (j <- samples.indices) {
            println("----- Sample " + j + " -----")
            println(samples(j))
            println()
          }
        }
      }
      iter.reset() //Сбросить итератор датасета перед новой эпохой
    }
    println("\n\nExample complete")
  }

  /** Загрузка примеров данных из текстов Шэкспира и сохранение их на диске
    *
    * @param miniBatchSize  Количество сегментов текста в каждом батче
    * @param sequenceLength Количество символов в каждом примере.
    */
  @throws[Exception]
  def getShakespeareIterator(miniBatchSize: Int, sequenceLength: Int): CharacterIterator = {
    //https://www.gutenberg.org/ebooks/100
    val url = "https://s3.amazonaws.com/dl4j-distribution/pg100.txt"
    val tempDir = System.getProperty("java.io.tmpdir")
    val fileLocation = tempDir + "/Shakespeare.txt" //Storage location from downloaded file
    val f = new File(fileLocation)
    if (!f.exists) {
      FileUtils.copyURLToFile(new URL(url), f)
      println("File downloaded to " + f.getAbsolutePath)
    } else {
      println("Using existing text file at " + f.getAbsolutePath)
    }
    if (!f.exists) throw new IOException("File does not exist: " + fileLocation) //Download problem?
    val validCharacters: Array[Char] = CharacterIterator.getMinimalCharacterSet //Which characters are allowed? Others will be removed
    new CharacterIterator(fileLocation, Charset.forName("UTF-8"), miniBatchSize, sequenceLength, validCharacters, new Random(12345))
  }

  /** Генерация тестового примера для обученой сети
    *
    * @param _initialization     String, может быть null, если null - то случайны символ
    * @param charactersToSample Размер тестового примера
    * @param net                MultiLayerNetwork с GravesLSTM/RNN слоями и softmax выходом
    * @param iter               CharacterIterator. Переходник между символами и числами
    */
  private def sampleCharactersFromNetwork(_initialization: String, net: MultiLayerNetwork, iter: CharacterIterator, rng: Random, charactersToSample: Int, numSamples: Int): Array[String] = {
    val initialization = if (_initialization == null) {
      String.valueOf(iter.getRandomCharacter)
    } else _initialization
    val initializationInput = Nd4j.zeros(numSamples, iter.inputColumns, initialization.length)
    val init = initialization.toCharArray
    for (i <- init.indices) {
      val idx = iter.convertCharacterToIndex(init(i))
      for (j <- 0 until numSamples) {
        initializationInput.putScalar(Array[Int](j, idx, i), 1.0f)
      }
    }
    val sb = mutable.ArrayBuffer.empty[StringBuilder]
    (0 until numSamples).foreach { _ =>
      sb += new StringBuilder(initialization)
    }

    //Получить один символ от сети и отдать его обратно
    //Полчение символов параллельно
    net.rnnClearPreviousState()
    var output = net.rnnTimeStep(initializationInput)
    output = output.tensorAlongDimension(output.size(2) - 1, 1, 0) //Gets the last time step output

    for (i <- 0 until charactersToSample) {
      //Поставить следущий вход получив предыдущий выход
      val nextInput = Nd4j.zeros(numSamples, iter.inputColumns)
      //Выход - это вероятностное распределение, взять лучшие и передать их дальше
      for (s <- 0 until numSamples) {
        val outputProbDistribution = new Array[Double](iter.totalOutcomes)
        for (j <- outputProbDistribution.indices) {
          outputProbDistribution(j) = output.getDouble(s, j)
        }
        val sampledCharacterIdx = sampleFromDistribution(outputProbDistribution, rng)
        nextInput.putScalar(Array[Int](s, sampledCharacterIdx), 1.0f)
        sb(s).append(iter.convertIndexToCharacter(sampledCharacterIdx)) //Добавить символ к итоговому результату
      }
      output = net.rnnTimeStep(nextInput) //Сделать временной шаг сети
    }
    val out = new Array[String](numSamples)
    for (i <- 0 until numSamples) {
      out(i) = sb(i).toString
    }
    out
  }

  def sampleFromDistribution(distribution: Array[Double], rng: Random): Int = {
    val d = rng.nextDouble
    val i = distribution
      .toIterator
      .scanLeft(0.0)({ case (acc, p) => acc + p })
      .drop(1)
      .indexWhere(_ >= d)
    if (i >= 0) {
      i
    } else {
      throw new IllegalArgumentException("Distribution is invalid? d=" + d + ", sum=" + distribution.sum)
    }
  }

}
