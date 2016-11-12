package schedule;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

import prescreening.DiagnosedPatient;

public class TimeSlot {
	private LocalDateTime _start = null;
	private LocalDateTime _end = null;
	private DiagnosedPatient _patient = null;
	
	public TimeSlot() {	
	}
	
	public void setStart(LocalDateTime start) {
		_start = start;
	}
	
	public LocalDateTime getStart() {
		return _start;
	}
	
	public void setEnd(LocalDateTime end) {
		_end = end;
	}
	
	public boolean isTooShort(long hours, long minutes) {
		if (_start.equals(_end)) {
			return true;
		}
		
		LocalDateTime tempEnd = _start.plusHours(hours);
		tempEnd = tempEnd.plusMinutes(minutes);
		return tempEnd.isAfter(_end);
	}
	
	public boolean isTooShort(long minutes) {
		LocalDateTime tempEnd = _start.plusMinutes(minutes);
		return tempEnd.isAfter(_end);
	}
	
	public LocalDateTime getEnd() {
		return _end;
	}
	
	public boolean isFreeSlot() {
		return _patient == null;
	}
	
	public void setPatient(DiagnosedPatient patient) {
		_patient = patient;
	}
	
	public DiagnosedPatient getPatient() {
		return _patient;
	}
	
	public String getTimeSlotDetail() {
		DateTimeFormatter formatter = DateTimeFormatter.ofPattern("dd-MM-yyyy (EE) HH:mm");
		String startString = _start.format(formatter);
		String endString = _end.format(formatter);
		if (isFreeSlot()) {
			return "Free: " + startString + " " + endString;
		} 
		return _patient.getName() + " : " + startString + " " + endString;
			
	}
}
