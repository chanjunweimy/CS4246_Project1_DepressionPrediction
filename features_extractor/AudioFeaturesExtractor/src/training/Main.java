package training;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Vector;

import features.Energy;
import features.MFCC;
import features.MagnitudeSpectrum;
import features.ZeroCrossing;
import signal.WaveIO;
import storage.ConfigFileHandler;

public class Main {
	private static final String FEATURE_MFCC = "MFCC";
	private static final String FEATURE_MS = "MS";
	private static final String FEATURE_ENERGY = "ENERGY";
	private static final String FEATURE_ZERO_CROSSING = "ZC";
	private static final String COMMAND_FEATURE = "FEATURE";
	private static final String COMMAND_GENERATE_CONFIG = "CONFIG";
	
	private boolean _hasMfcc = false;
	private boolean _hasMs = false;
	private boolean _hasEnergy = false;
	private boolean _hasZeroCrossing = false;
	
	public Main() {
	}
	
	public void execute(String[] args) {
		if (args.length == 0) {
			System.out.println("No arguments are passed in! Please specify if you want \"FEATURE\" or \"CONFIG\"");
		} else if (args.length > 0) {
			args[0] = args[0].trim().toUpperCase();
			if (COMMAND_GENERATE_CONFIG.equals(args[0])) {
				generateConfigFile();
			} else if (COMMAND_FEATURE.equals(args[0])) {
				generateFeatureFile(args);
			}
		}
	}
	
	private void generateConfigFile() {
		ConfigFileHandler.getInstance().generateConfigFile();
	}
	
	private void generateFeatureFile(String[] args) {
		specifyFeatures(args);
		
		ConfigFileHandler config = ConfigFileHandler.getInstance();
		String biasFile = config.getOutputBiasFileName();
		String unbiasFile = config.getOutputFileName();
		String inputFilePathString = config.getInputFilePath();
		File inputFilePath = new File(inputFilePathString);
		File[] inputFiles = inputFilePath.listFiles();
		computeMfccMsEnergyAndZcBiasAndUnbias(inputFiles, biasFile, unbiasFile);
	}
	
	private boolean computeMfccMsEnergyAndZcBiasAndUnbias(File[] audioFiles, String biasFile, String unbiasFile) {
		writeToFile(unbiasFile, false, "");
		writeToFile(biasFile, false, "");
		
		Vector<String> features = retrieveFeatures(audioFiles);
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

	/**
	 * @param audioFiles
	 * @return
	 */
	private Vector<String> retrieveFeatures(File[] audioFiles) {
		Vector <String> features = new Vector <String>();
		
		
		for (int i = 0; i < audioFiles.length; i++) {
			String audioName = audioFiles[i].getAbsolutePath();
			
			WaveIO waveio = new WaveIO();
			short[] signal = waveio.readWave(audioName);

			StringBuffer buffer = new StringBuffer();
			
			boolean hasStarted = false;
			
			if (_hasMfcc) {
				MFCC mfcc = new MFCC();
				mfcc.process(signal);
				double[] mean = mfcc.getMeanFeature();
				buffer.append(mean[0]);
				for (int j = 1; j < mean.length; j++) {
					buffer.append(", "); 
					buffer.append(mean[j]);
				}
				hasStarted = true;
			}
			
			if (_hasMs) {
				MagnitudeSpectrum ms = new MagnitudeSpectrum();
				double[] meanMs = ms.getFeature(signal);
				
				int msStart = 0;
				if (!hasStarted) {
					msStart = 1;
					buffer.append(meanMs[0]);
				}
				
				for (int j = msStart; j < meanMs.length; j++) {
					buffer.append(", "); 
					buffer.append(meanMs[j]);
				}
				hasStarted = true;
			}
			
			if (_hasEnergy) {
				Energy energy = new Energy();
				double[] meanEnergy = energy.getFeature(signal);
				
				int energyStart = 0;
				if (!hasStarted) {
					energyStart = 1;
					buffer.append(meanEnergy[0]);
				}
				
				for (int j = energyStart; j < meanEnergy.length; j++) {
					buffer.append(", "); 
					buffer.append(meanEnergy[j]);
				}
				hasStarted = true;
			}
			
			if (_hasZeroCrossing) {
				ZeroCrossing zc = new ZeroCrossing();
				double[] meanZc = zc.getFeature(signal);
				
				int zcStart = 0;
				if (!hasStarted) {
					zcStart = 1;
					buffer.append(meanZc[0]);
				}
				
				for (int j = zcStart; j < meanZc.length; j++) {
					buffer.append(", "); 
					buffer.append(meanZc[j]);
				}
				hasStarted = true;
			}
			
			
			buffer.append("\n");
			features.add(buffer.toString());
		}
		return features;
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

	/**
	 * @param args
	 */
	private void specifyFeatures(String[] args) {
		for (int i = 1; i < args.length; i++) {
			args[i] = args[i].trim().toUpperCase();
			if (FEATURE_MFCC.equals(args[i])) {
				_hasMfcc = true;
			} else if (FEATURE_MS.equals(args[i])) {
				_hasMs = true;
			} else if (FEATURE_ENERGY.equals(args[i])) {
				_hasEnergy = true;
			} else if (FEATURE_ZERO_CROSSING.equals(args[i])) {
				_hasZeroCrossing = true;
			}
		}
	}
	
	public static void main(String[] args) {
		Main prog = new Main();
		prog.execute(args);
	}
}
