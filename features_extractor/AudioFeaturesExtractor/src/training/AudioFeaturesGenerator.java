package training;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import signal.WaveIO;
import features.Energy;
import features.MFCC;
import features.MagnitudeSpectrum;
import features.ZeroCrossing;

public class AudioFeaturesGenerator {
	public static final String EXT_TXT = ".txt";
	public static final String EXT_WAV = ".wav";

	//public static final String FILEPATH_FEATURE_OUT = "data/features/";
	public static final String FILEPATH_FEATURE_OUT = "";
	public static final String FILEPATH_DCAPSWOZ = "data/dcapswoz_audio/dev";

	public static final String EMOTION_DCAPSWOZ_MFCC = FILEPATH_FEATURE_OUT + "emotion_dcapswoz_mfcc.txt";
	public static final String EMOTION_DCAPSWOZ_SPECTRUM = FILEPATH_FEATURE_OUT + "emotion_dcapswoz_spectrum.txt";

	public boolean computeMfccMsAndZc(File[] audioFiles, String filename) {
		writeToFile(filename, false, "");
		
		try (FileWriter fw = new FileWriter(filename, true)){
			for (int i = 0; i < audioFiles.length; i++) {
				String audioName = audioFiles[i].getAbsolutePath();
				
				WaveIO waveio = new WaveIO();
				short[] signal = waveio.readWave(audioName);
				
				MFCC mfcc = new MFCC();
				mfcc.process(signal);
				double[] mean = mfcc.getMeanFeature();
				
				MagnitudeSpectrum ms = new MagnitudeSpectrum();
				double[] meanMs = ms.getFeature(signal);
				
				ZeroCrossing zc = new ZeroCrossing();
				double[] meanZc = zc.getFeature(signal);
				
				StringBuffer buffer = new StringBuffer();
				
				//buffer.append(audioFiles[i].getName());
				if (mean.length > 0) {
					buffer.append(mean[0]);
				} else if (meanMs.length > 0) {
					buffer.append(meanMs[0]);
				} else if (meanZc.length > 0) {
					buffer.append(meanZc[0]);				
				}
				
				for (int j = 1; j < mean.length; j++) {
					buffer.append(", "); 
					buffer.append(mean[j]);
				}
				
				for (int j = 0; j < meanMs.length; j++) {
					buffer.append(", "); 
					buffer.append(meanMs[j]);
				}
				
				for (int j = 0; j < meanZc.length; j++) {
					buffer.append(", "); 
					buffer.append(meanZc[j]);
				}
				
				buffer.append("\n");
				fw.write(buffer.toString());
			}
			fw.close();
		} catch (IOException e) {
			e.printStackTrace();
			return false;
		}
		
		return true;
	}
	
	public boolean computeMFCCAndMagnitudeSpectrum(File[] audioFiles, String filename) {
		writeToFile(filename, false, "");
		
		try (FileWriter fw = new FileWriter(filename, true)){
			for (int i = 0; i < audioFiles.length; i++) {
				String audioName = audioFiles[i].getAbsolutePath();
				
				if (audioName.endsWith("null.wav")) {
					continue;
				}
				WaveIO waveio = new WaveIO();
				short[] signal = waveio.readWave(audioName);
				
				MFCC mfcc = new MFCC();
				mfcc.process(signal);
				double[] mean = mfcc.getMeanFeature();
				
				MagnitudeSpectrum ms = new MagnitudeSpectrum();
				double[] meanMs = ms.getFeature(signal);
				
				StringBuffer buffer = new StringBuffer();
				
				//buffer.append(audioFiles[i].getName());
				if (mean.length > 0) {
					buffer.append(mean[0]);
				} else if (meanMs.length > 0) {
					buffer.append(meanMs[0]);
				}
				
				for (int j = 1; j < mean.length; j++) {
					buffer.append(", "); 
					buffer.append(mean[j]);
				}
				
				for (int j = 0; j < meanMs.length; j++) {
					buffer.append(", "); 
					buffer.append(meanMs[j]);
				}
				
				buffer.append("\n");
				fw.write(buffer.toString());
			}
			fw.close();
		} catch (IOException e) {
			e.printStackTrace();
			return false;
		}
		
		return true;
	}
	
	public boolean computeMFCC(File[] audioFiles, String filename) {
		writeToFile(filename, false, "");
		for (int i = 0; i < audioFiles.length; i++) {
			String audioName = audioFiles[i].getAbsolutePath();
			
			if (audioName.endsWith("null.wav")) {
				continue;
			}
			WaveIO waveio = new WaveIO();
			short[] signal = waveio.readWave(audioName);
			
			MFCC mfcc = new MFCC();
			mfcc.process(signal);
			double[] mean = mfcc.getMeanFeature();
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
	
	public boolean computeEnergy(File[] audioFiles, String filename) {
		writeToFile(filename, false, "");
		for (int i = 0; i < audioFiles.length; i++) {
			String audioName = audioFiles[i].getAbsolutePath();
			WaveIO waveio = new WaveIO();
			short[] signal = waveio.readWave(audioName);
			
			Energy energy1 = new Energy();
			double[] feature = energy1.getFeature(signal);
			StringBuffer buffer = new StringBuffer();
			
			//buffer.append(audioFiles[i].getName());
			if (feature.length > 0) {
				buffer.append(feature[0]);
			}
			
			for (int j = 1; j < feature.length; j++) {
				buffer.append(", "); 
				buffer.append(feature[j]);
			}
			buffer.append("\n");
			if (!writeToFile(filename, true, buffer.toString())) {
				return false;
			}
		}
		return true;
	}
	
	public boolean computeMagnitudeSpectrum(File[] audioFiles, String filename) {
		writeToFile(filename, false, "");
		for (int i = 0; i < audioFiles.length; i++) {
			String audioName = audioFiles[i].getAbsolutePath();
			WaveIO waveio = new WaveIO();
			short[] signal = waveio.readWave(audioName);
			
			MagnitudeSpectrum spectrum = new MagnitudeSpectrum();
			double[] feature = spectrum.getFeature(signal);
			StringBuffer buffer = new StringBuffer();
			
			if (feature.length > 0) {
				buffer.append(feature[0]);
			}
			
			for (int j = 1; j < feature.length; j++) {
				buffer.append(", "); 
				buffer.append(feature[j]);
			}
			buffer.append("\n");
			if (!writeToFile(filename, true, buffer.toString())) {
				return false;
			}
		}
		return true;
	}
	
	public boolean computeZeroCrossing(File[] audioFiles, String filename) {
		writeToFile(filename, false, "");
		for (int i = 0; i < audioFiles.length; i++) {
			String audioName = audioFiles[i].getAbsolutePath();
			WaveIO waveio = new WaveIO();
			short[] signal = waveio.readWave(audioName);
			
			ZeroCrossing zc = new ZeroCrossing();
			double[] feature = zc.getFeature(signal);
			StringBuffer buffer = new StringBuffer();
			
			if (feature.length > 0) {
				buffer.append(feature[0]);
			}
			
			for (int j = 1; j < feature.length; j++) {
				buffer.append(", ");  
				buffer.append(feature[j]);
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
		
		File emotionMfccFile = featureGenerator.createFile(AudioFeaturesGenerator.EMOTION_DCAPSWOZ_MFCC);
		//File emotionSpectrumFile = featureGenerator.createFile(AudioFeaturesGenerator.EMOTION_DCAPSWOZ_SPECTRUM);
		

		if (!featureGenerator.computeMFCCAndMagnitudeSpectrum(emotionFiles, emotionMfccFile.getAbsolutePath())) {
			System.exit(-1);
		} /* else if (!featureGenerator.computeMagnitudeSpectrum(emotionFiles, emotionSpectrumFile.getAbsolutePath())) {
			System.exit(-1);
		}*/
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
