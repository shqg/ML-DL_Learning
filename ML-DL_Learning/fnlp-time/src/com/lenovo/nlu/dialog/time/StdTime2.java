package com.lenovo.nlu.dialog.time;

import java.util.ArrayList;
import java.util.List;

/**
 * @author gengsq2
 * 
 */
public class StdTime2 {

	/**
	 * @param text
	 * @return 句子中匹配到的时间，并标准化返回
	 */
	public static List<TimeFormat2> normalTime(String text) {

		String path = TimeNormalizer.class.getResource("").getPath();
		String classPath = path.substring(0, path.length() - 1);
		TimeNormalizer normalizer = new TimeNormalizer(classPath + "/TimeExp2.m");
		normalizer.parse(text);// 抽取时间
		TimeUnit[] unit = normalizer.getTimeUnit();
		System.out.println(text);

		List<TimeFormat2> ltf = new ArrayList<TimeFormat2>();
//		for (int i = 0; i < unit.length; i++) {
//			System.out.println(DateUtil.formatDateDefault(unit[i].getTime()) + " " + unit[i].getIsAllDayTime());
//		}
		if (unit.length > 2) {			
			for (int i = 0; i < unit.length; i++) {
				if (i + 2 < unit.length) {
					String s1 = DateUtil.formatDateDefault(unit[i].getTime()) + " " + unit[i].getIsAllDayTime();
//					System.out.println("s1:" + unit[i].Time_Expression);
					String s2 = DateUtil.formatDateDefault(unit[i + 1].getTime()) + " " + unit[i + 1].getIsAllDayTime();
					String s3 = DateUtil.formatDateDefault(unit[i + 2].getTime()) + " " + unit[i + 2].getIsAllDayTime();
//					System.out.println("s2:" + unit[i+1].Time_Expression);
//					System.out.println("s3:" + unit[i+2].Time_Expression);
					if (s1.equals(s2)) {
						TimeFormat2 tf = new TimeFormat2(null, null, null);
						tf.setStartTime(s1);
						tf.setEndTime(s3);
						ltf.add(tf);
					} else {
						TimeFormat2 tf = new TimeFormat2(null, null, null);
						tf.setStartTime(s1);
						tf.setEndTime(s1);
						tf.setTimeExp(unit[i].Time_Expression);
						ltf.add(tf);
						tf = new TimeFormat2(null, null, null);
						tf.setStartTime(s2);
						tf.setEndTime(s2);
						tf.setTimeExp(unit[i+1].Time_Expression);
						ltf.add(tf);
						tf = new TimeFormat2(null, null, null);
						tf.setStartTime(s3);
						tf.setEndTime(s3);
						tf.setTimeExp(unit[i+2].Time_Expression);
						ltf.add(tf);
					}
					i = i + 2;
				} else {
					TimeFormat2 tf = new TimeFormat2(null, null, null);
					String s5 = DateUtil.formatDateDefault(unit[i].getTime()) + " " + unit[i].getIsAllDayTime();
//					System.out.println("s5:" + unit[i].Time_Expression);
					tf.setStartTime(s5);
					tf.setEndTime(s5);
					tf.setTimeExp(unit[i].Time_Expression);
					ltf.add(tf);
				}
			}
		} else {
			for (int i = 0; i < unit.length; i++) {
				String s4 = DateUtil.formatDateDefault(unit[i].getTime()) + " " + unit[i].getIsAllDayTime();
//				System.out.println("s4:" + unit[i].Time_Expression);
				TimeFormat2 tf = new TimeFormat2(null, null, null);
				tf.setStartTime(s4);
				tf.setEndTime(s4);
				tf.setTimeExp(unit[i].Time_Expression);
				ltf.add(tf);
			}

		}

		return ltf;
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		List<TimeFormat2> ltf = normalTime("今明天的天气的天气");
//		System.out.println("ltf.size():" + ltf.size());
		for (TimeFormat2 t : ltf) {
			System.out.println(t.toString());
		}

	}

}
