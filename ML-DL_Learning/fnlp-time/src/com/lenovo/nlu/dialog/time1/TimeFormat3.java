package com.lenovo.nlu.dialog.time1;

import java.util.List;

public class TimeFormat3 {
	private  List<String> Time;
    private String timeExp;
	
    public TimeFormat3(String timeExp,List<String> Time ) {
		super();
		this.Time =  Time;
		this.timeExp = timeExp;
	}

	public List<String> getTime() {
		return Time;
	}

	public void setTime(List<String> timeExp) {
		this.Time = timeExp;
	}
	public String TimeExp() {
		return timeExp;
	}

	public void setTimeExp(String timeExp) {
		this.timeExp = timeExp;
	}

	public String toString() {
		String str="";
		for(String s:Time){
			str=str+"\r"+s;
		}
		str=str+"-"+timeExp;
		return str;
	}

	 

}
