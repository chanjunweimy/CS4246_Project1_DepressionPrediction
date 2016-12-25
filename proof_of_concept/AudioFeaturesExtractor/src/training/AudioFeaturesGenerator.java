package training;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
//import java.util.Vector;

import com.opencsv.CSVReader;

import signal.WaveIO;
//import features.Energy;
import features.MFCC;
//import features.MagnitudeSpectrum;
//import features.ZeroCrossing;

public class AudioFeaturesGenerator {
	public static final String EXT_TXT = ".txt";
	public static final String EXT_WAV = ".wav";

	//public static final String FILEPATH_FEATURE_OUT = "data/features/";
	public static final String FILEPATH_FEATURE_OUT = "";
	
	//public static final String FILEPATH_DCAPSWOZ = "data/dcapswoz_audio_participantonly_merged/dev";
		public static final String FILEPATH_DCAPSWOZ = "data/dcapswoz_audio_participantonly_merged/train";

	//public static final String FILEPATH_RESULT = "data/dcapswoz_audio_participantonly/result/dev_split.csv";
	public static final String FILEPATH_RESULT = "data/dcapswoz_audio_participantonly/result/training_split.csv";
	
	public static final String EMOTION_DCAPSWOZ_MFCC = FILEPATH_FEATURE_OUT + "emotion_dcapswoz_mfcc.txt";
	public static final String EMOTION_DCAPSWOZ_SPECTRUM = FILEPATH_FEATURE_OUT + "emotion_dcapswoz_spectrum.txt";
	public static final String EMOTION_DCAPSWOZ_ALL = FILEPATH_FEATURE_OUT + "emotion_dcapswoz_all.txt";
	public static final String EMOTION_DCAPSWOZ_ALL_BIAS = FILEPATH_FEATURE_OUT + "emotion_dcapswoz_all_bias.txt";
	
	public static final String OUTCOME_PHQ8 = FILEPATH_FEATURE_OUT + "y.txt";
	public static final String OUTCOME_BINARY = FILEPATH_FEATURE_OUT + "y_bin.txt";

	private HashMap<Integer, Integer> _mapBinary = null;
	private HashMap<Integer, Integer> _mapPHQ8 = null;
	
	private ArrayList<Integer> _outcomeBinary = null;
	private ArrayList<Integer> _outcomePHQ8 = null;
	
	public AudioFeaturesGenerator() {
		_mapBinary = new HashMap<Integer, Integer>();
		_mapPHQ8 = new HashMap<Integer, Integer>();
		
		_outcomeBinary = new ArrayList<Integer>();
		_outcomePHQ8 = new ArrayList<Integer>();
	}
	
	
	private void saveOutcomes(String filename, ArrayList<Integer> outcomes) {
		writeToFile(filename, false, "");
		
		StringBuilder builder = new StringBuilder();
		
		for (Integer outcome : outcomes) {
			builder.append(outcome);
			builder.append(System.lineSeparator());
		}
		writeToFile(filename, true, builder.toString());
	}
	
	public boolean computeMFCC(File[] audioFiles, String filename) {
		writeToFile(filename, false, "");
		for (int i = 0; i < audioFiles.length; i++) {
			String audioName = audioFiles[i].getAbsolutePath();
			
			String name = audioFiles[i].getName();
			int id = Integer.parseInt(name.split("_")[0]);
			_outcomeBinary.add(_mapBinary.get(id));
			_outcomePHQ8.add(_mapPHQ8.get(id));
			
			if (audioName.endsWith("null.wav")) {
				continue;
			}
			WaveIO waveio = new WaveIO();
			short[] signal = waveio.readWave(audioName);
			
			MFCC mfcc = new MFCC();
			mfcc.process(signal);
			double[] mean = mfcc.getMeanFeature();
			
			//System.out.println(mean.length);
			
			StringBuffer buffer = new StringBuffer();
			
			//buffer.append(audioFiles[i].getName());
			if (mean.length > 0) {
				buffer.append(mean[0]);
			}
			
			for (int j = 1; j < mean.length; j++) {
				buffer.append(", "); 
				buffer.append(mean[j]);
			}
			buffer.append("\n");
			if (!writeToFile(filename, true, buffer.toString())) {
				return false;
			}
		}
		return true;
	}
	
	private boolean writeToFile(String filename, boolean isAppend, String line) {
		try (FileWriter fw = new FileWriter(filename, isAppend)){
			fw.write(line);
			fw.close();
		} catch (IOException e) {
			e.printStackTrace();
			return false;
		}
		 
		return true;
	}
	
	public File createFile(String filepath) {
		File f = new File(filepath);
		//f.getParentFile().mkdirs(); 
		try {
			f.createNewFile();
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
		return f;
	}
	
	private static void trainDcapswozEmotion(AudioFeaturesGenerator featureGenerator) {
		trainDcapswozEmotion(featureGenerator, FILEPATH_DCAPSWOZ);
	}
	
	private static void trainDcapswozEmotion(AudioFeaturesGenerator featureGenerator, String trainFilePath) {
		File emotionTrain = new File(trainFilePath);
		File[] emotionFiles = emotionTrain.listFiles();
		
		//File emotionAllFeaturesFile = featureGenerator.createFile(AudioFeaturesGenerator.EMOTION_DCAPSWOZ_ALL);
		//File emotionBiasAllFeaturesFile = featureGenerator.createFile(AudioFeaturesGenerator.EMOTION_DCAPSWOZ_ALL_BIAS);
		File emotionMfccFile = featureGenerator.createFile(AudioFeaturesGenerator.EMOTION_DCAPSWOZ_MFCC);
		//File emotionSpectrumFile = featureGenerator.createFile(AudioFeaturesGenerator.EMOTION_DCAPSWOZ_SPECTRUM);
		
		featureGenerator.initializeMaps();
		
		if (!featureGenerator.computeMFCC(emotionFiles,
				emotionMfccFile.getAbsolutePath())) {
			System.exit(-1);
		}	
		featureGenerator.saveOutcomes(OUTCOME_PHQ8, featureGenerator._outcomePHQ8);
		featureGenerator.saveOutcomes(OUTCOME_BINARY, featureGenerator._outcomeBinary);
		
		/*
		if (!featureGenerator.computeMfccMsEnergyAndZcBiasAndUnbias(emotionFiles, 
																    emotionBiasAllFeaturesFile.getAbsolutePath(),
																    emotionAllFeaturesFile.getAbsolutePath())) {
			System.exit(-1);
		}*/
			/* 
		if (!featureGenerator.computeMfccMsEnergyAndZcBias(emotionFiles, emotionAllFeaturesFile.getAbsolutePath())) {
			System.exit(-1);
		} else if (!featureGenerator.computeMagnitudeSpectrum(emotionFiles, emotionSpectrumFile.getAbsolutePath())) {
			System.exit(-1);
		}*/
	}
	
	
	private void initializeMaps() {
		CSVReader reader = null;
        try {
            reader = new CSVReader(new FileReader(FILEPATH_RESULT));
            String[] line;
            while ((line = reader.readNext()) != null) {
            	if (line[0].toLowerCase().equals("participant_id")) {
            		continue;
            	}
            	
                int id = Integer.parseInt(line[0]);
                int bin = Integer.parseInt(line[1]);
                int phq8 = Integer.parseInt(line[2]);
                
                _mapBinary.put(id, bin);
                _mapPHQ8.put(id, phq8);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
	}



	public static void main(String[] args) {
		AudioFeaturesGenerator featureGenerator = new AudioFeaturesGenerator();
		
		if (args.length == 0) {
			trainDcapswozEmotion(featureGenerator);
		} else if (args.length > 0) {
			trainDcapswozEmotion(featureGenerator, args[0]);
		}
		
	}
	

}
