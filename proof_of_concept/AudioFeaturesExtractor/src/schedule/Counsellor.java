package schedule;

import java.util.ArrayList;

public class Counsellor {
	private ArrayList <TimeSlot> _timeSlots = null;
	private String _name = null;
	
	public Counsellor() {	
		_timeSlots = new ArrayList<TimeSlot>();
	}
	
	public void addTimeSlot(TimeSlot timeSlot) {
		_timeSlots.add(timeSlot);
	}
	
	public ArrayList <TimeSlot> getTimeSlots() {
		return _timeSlots;
	}
	
	public void setName(String name) {
		_name = name;
	}
	
	public String getName() {
		return _name;
	}
	
	public void printCounsellor() {
		System.out.println(_name + ":");
		for (TimeSlot timeSlot: _timeSlots) {
			System.out.println(timeSlot);
		}
	}
}
