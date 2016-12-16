package noiseremoval;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.Map;

public class NoiseRemover {
	private static NoiseRemover _remover = null;
	private static final String NOISE_SAMPLE = "noiseaud.wav";
	private static final String NOISE_PROFILE = "noise.prof";
	
	private NoiseRemover() {
	}
	
	
	public static NoiseRemover getInstance() {
		if (_remover == null) {
			_remover = new NoiseRemover();
		}
		return _remover;
	}
	
	public void removeNoise(String audioOriginal, String audioCleaned) {
		generateSampleNoise(audioOriginal);
		generateNoiseProfile();
		cleanNoiseSample(audioOriginal, audioCleaned);
	}
	
	@SuppressWarnings("unused")
	private void printMap(Map<String, String> map) {
		System.out.print(Arrays.toString(map.entrySet().toArray()));
	}
	
	private void generateSampleNoise(String audioOriginal) {
		Process process;
		try {
			String command = "ffmpeg -i " + audioOriginal + " -vn -ss 00:00:00 -t 00:00:01 " + NOISE_SAMPLE;
			System.out.println(command);
			process = Runtime.getRuntime().exec(command);
			//process = pb.start();
			InputStream is = process.getInputStream();
			InputStreamReader isr = new InputStreamReader(is);
			BufferedReader br = new BufferedReader(isr);
			String line;
			
			while ((line = br.readLine()) != null) {
				 line = line.trim();
				 if (line.startsWith("Error:")) {
					 System.out.println(line);
				 } else {
					 System.out.println(line);
				 }
			}
			br.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		System.out.println("generated Sample Noise");
	}
	
	private void generateNoiseProfile() {
		Process process;
		try {
			//ProcessBuilder pb = new ProcessBuilder("sox", NOISE_SAMPLE, "-n", "noiseprof", NOISE_PROFILE);
			//process = pb.start();
			String command = "sox " + NOISE_SAMPLE + " -n noiseprof " + NOISE_PROFILE;
			System.out.println(command);
			process = Runtime.getRuntime().exec(command);
			InputStream is = process.getInputStream();
			InputStreamReader isr = new InputStreamReader(is);
			BufferedReader br = new BufferedReader(isr);
			String line;
			
			while ((line = br.readLine()) != null) {
				 line = line.trim();
				 if (line.startsWith("Error:")) {
					 System.out.println(line);
				 } else {
					 System.out.println(line);
				 }
			}
			br.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		System.out.println("generated Noise Profile");
	}
	
	private void cleanNoiseSample(String audioOriginal, String audioCleaned) {
		Process process;
		try {
			//ProcessBuilder pb = new ProcessBuilder("sox", audioOriginal, audioCleaned, "noisered", NOISE_PROFILE, "0.21");
			//process = pb.start();
			String command = "sox " + audioOriginal + " " + audioCleaned + " noisered " + NOISE_PROFILE + " 0.21";
			System.out.println(command);
			process = Runtime.getRuntime().exec(command);
			InputStream is = process.getInputStream();
			InputStreamReader isr = new InputStreamReader(is);
			BufferedReader br = new BufferedReader(isr);
			String line;
			
			while ((line = br.readLine()) != null) {
				 line = line.trim();
				 if (line.startsWith("Error:")) {
					 System.out.println(line);
				 } else {
					 System.out.println(line);
				 }
			}
			br.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		System.out.println("cleaned Noise Sample");
	}
	
	public static void main(String[] args) {
		NoiseRemover.getInstance().removeNoise("test.wav", "clean.wav");
	}
}
