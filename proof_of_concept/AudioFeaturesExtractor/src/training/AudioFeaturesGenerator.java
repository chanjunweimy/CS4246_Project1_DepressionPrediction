package training;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Vector;

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
	//public static final String FILEPATH_DCAPSWOZ = "data/dcapswoz_audio/train";

	public static final String EMOTION_DCAPSWOZ_MFCC = FILEPATH_FEATURE_OUT + "emotion_dcapswoz_mfcc.txt";
	public static final String EMOTION_DCAPSWOZ_SPECTRUM = FILEPATH_FEATURE_OUT + "emotion_dcapswoz_spectrum.txt";
	public static final String EMOTION_DCAPSWOZ_ALL = FILEPATH_FEATURE_OUT + "emotion_dcapswoz_all.txt";
	public static final String EMOTION_DCAPSWOZ_ALL_BIAS = FILEPATH_FEATURE_OUT + "emotion_dcapswoz_all_bias.txt";

	
	public AudioFeaturesGenerator() {
	}
	

	
	public boolean computeMfccMsEnergyAndZcBiasAndUnbias(File[] audioFiles, String biasFile, String unbiasFile) {
		writeToFile(unbiasFile, false, "");
		writeToFile(biasFile, false, "");
		
		int mfccLength = -1;
		int msLength = -1;
		int energyLength = -1;
		int zcLength = -1;
		Vector <String> features = new Vector <String>();
		
		
		for (int i = 0; i < audioFiles.length; i++) {
			String audioName = audioFiles[i].getAbsolutePath();
			
			WaveIO waveio = new WaveIO();
			short[] signal = waveio.readWave(audioName);
			
			MFCC mfcc = new MFCC();
			mfcc.process(signal);
			double[] mean = mfcc.getMeanFeature();
			
			MagnitudeSpectrum ms = new MagnitudeSpectrum();
			double[] meanMs = ms.getFeature(signal);
			
			Energy energy = new Energy();
			double[] meanEnergy = energy.getFeature(signal);
			
			ZeroCrossing zc = new ZeroCrossing();
			double[] meanZc = zc.getFeature(signal);
			
			StringBuffer buffer = new StringBuffer();
			
			//buffer.append(audioFiles[i].getName());
			buffer.append(mean[0]);
			
			if (mfccLength == -1) {
				mfccLength = mean.length;
			} else if (mfccLength != mean.length) {
				System.err.println("MFCC has error");
			}
			
			if (msLength == -1) {
				msLength = meanMs.length;
			} else if (msLength != meanMs.length) {
				System.err.println("MS has error");
			}
			
			if (energyLength == -1) {
				energyLength = meanEnergy.length;
			} else if (energyLength != meanEnergy.length) {
				System.err.println("Energy has error");
			}
			
			if (zcLength == -1) {
				zcLength = meanZc.length;
			} else if (zcLength != meanZc.length) {
				System.err.println("ZC has error");
			}
			
			for (int j = 1; j < mean.length; j++) {
				buffer.append(", "); 
				buffer.append(mean[j]);
			}
			
			for (int j = 0; j < meanMs.length; j++) {
				buffer.append(", "); 
				buffer.append(meanMs[j]);
			}
			
			for (int j = 0; j < meanEnergy.length; j++) {
				buffer.append(", "); 
				buffer.append(meanEnergy[j]);
			}
			
			for (int j = 0; j < meanZc.length; j++) {
				buffer.append(", "); 
				buffer.append(meanZc[j]);
			}
			
			buffer.append("\n");
			features.add(buffer.toString());
		}
		try (
				FileWriter fw = new FileWriter(unbiasFile, true); 
				FileWriter biasFw = new FileWriter(biasFile, true);){
			
			for (String feature : features) {
				fw.write(feature);
				biasFw.write("1, " + feature);
			}
			
			fw.close();
			biasFw.close();
		} catch (IOException e) {
			e.printStackTrace();
			return false;
		}
		
		return true;
	}
	
	public boolean computeMfccMsEnergyAndZcBias(File[] audioFiles, String filename) {
		writeToFile(filename, false, "");
		
		int mfccLength = -1;
		int msLength = -1;
		int energyLength = -1;
		int zcLength = -1;
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
				
				Energy energy = new Energy();
				double[] meanEnergy = energy.getFeature(signal);
				
				ZeroCrossing zc = new ZeroCrossing();
				double[] meanZc = zc.getFeature(signal);
				
				StringBuffer buffer = new StringBuffer();
				
				//buffer.append(audioFiles[i].getName());
				buffer.append(1);
				
				if (mfccLength == -1) {
					mfccLength = mean.length;
				} else if (mfccLength != mean.length) {
					System.err.println("MFCC has error");
				}
				
				if (msLength == -1) {
					msLength = meanMs.length;
				} else if (msLength != meanMs.length) {
					System.err.println("MS has error");
				}
				
				if (energyLength == -1) {
					energyLength = meanEnergy.length;
				} else if (energyLength != meanEnergy.length) {
					System.err.println("Energy has error");
				}
				
				if (zcLength == -1) {
					zcLength = meanZc.length;
				} else if (zcLength != meanZc.length) {
					System.err.println("ZC has error");
				}
				
				for (int j = 0; j < mean.length; j++) {
					buffer.append(", "); 
					buffer.append(mean[j]);
				}
				
				for (int j = 0; j < meanMs.length; j++) {
					buffer.append(", "); 
					buffer.append(meanMs[j]);
				}
				
				for (int j = 0; j < meanEnergy.length; j++) {
					buffer.append(", "); 
					buffer.append(meanEnergy[j]);
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
	
	public boolean computeMfccMsEnergyAndZc(File[] audioFiles, String filename) {
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
				
				Energy energy = new Energy();
				double[] meanEnergy = energy.getFeature(signal);
				
				ZeroCrossing zc = new ZeroCrossing();
				double[] meanZc = zc.getFeature(signal);
				
				StringBuffer buffer = new StringBuffer();
				
				//buffer.append(audioFiles[i].getName());
				int meanStart = 0;
				int meanMsStart = 0;
				int meanEnergyStart = 0;
				int meanZcStart = 0;
				
				if (mean.length > 0) {
					buffer.append(mean[0]);
					meanStart = 1;
				} else if (meanMs.length > 0) {
					buffer.append(meanMs[0]);
					meanMsStart = 1;
				} else if (meanEnergy.length > 0) {
					buffer.append(meanEnergy[0]);
					meanEnergyStart = 1;
				} else if (meanZc.length > 0) {
					buffer.append(meanZc[0]);		
					meanZcStart = 1;
				}
				
				for (int j = meanStart; j < mean.length; j++) {
					buffer.append(", "); 
					buffer.append(mean[j]);
				}
				
				for (int j = meanMsStart; j < meanMs.length; j++) {
					buffer.append(", "); 
					buffer.append(meanMs[j]);
				}
				
				for (int j = meanEnergyStart; j < meanEnergy.length; j++) {
					buffer.append(", "); 
					buffer.append(meanEnergy[j]);
				}
				
				for (int j = meanZcStart; j < meanZc.length; j++) {
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
		
		File emotionAllFeaturesFile = featureGenerator.createFile(AudioFeaturesGenerator.EMOTION_DCAPSWOZ_ALL);
		File emotionBiasAllFeaturesFile = featureGenerator.createFile(AudioFeaturesGenerator.EMOTION_DCAPSWOZ_ALL_BIAS);
		//File emotionMfccFile = featureGenerator.createFile(AudioFeaturesGenerator.EMOTION_DCAPSWOZ_MFCC);
		//File emotionSpectrumFile = featureGenerator.createFile(AudioFeaturesGenerator.EMOTION_DCAPSWOZ_SPECTRUM);
		
		if (!featureGenerator.computeMFCC(emotionFiles,
			    emotionAllFeaturesFile.getAbsolutePath())) {
			System.exit(-1);
		}	
		
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
	
	
	public static void main(String[] args) {
		AudioFeaturesGenerator featureGenerator = new AudioFeaturesGenerator();
		
		if (args.length == 0) {
			trainDcapswozEmotion(featureGenerator);
		} else if (args.length > 0) {
			trainDcapswozEmotion(featureGenerator, args[0]);
		}
		
	}
	

}
