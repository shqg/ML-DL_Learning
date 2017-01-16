package com.lenovo.nlu.dialog.time1;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

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
		// TimeNormalizer normalizer = new TimeNormalizer(classPath +
		// "/TimeExp2.m");
		TimeNormalizer normalizer = new TimeNormalizer(
				"/home/gengsq2/java-neon/workspace2/fnlp-time/src/com/lenovo/nlu/dialog/time1/TimeExp2.m");
		normalizer.parse(text);// 抽取时间
		TimeUnit[] unit = normalizer.getTimeUnit();
		System.out.println(text);
		List<TimeFormat2> ltf = new ArrayList<TimeFormat2>();
		if (unit.length > 2) {
			for (int i = 0; i < unit.length; i++) {
				if (i + 2 < unit.length) {
					String s1 = DateUtil.formatDateDefault(unit[i].getTime()) + " " + unit[i].getIsAllDayTime();
					// System.out.println("s1:" + unit[i].Time_Expression+s1);
					String s2 = DateUtil.formatDateDefault(unit[i + 1].getTime()) + " " + unit[i + 1].getIsAllDayTime();
					String s3 = DateUtil.formatDateDefault(unit[i + 2].getTime()) + " " + unit[i + 2].getIsAllDayTime();
					// System.out.println("s2:" + unit[i+1].Time_Expression+s2);
					// System.out.println("s3:" + unit[i+2].Time_Expression+s3);
					if (s1.equals(s2)) {
						TimeFormat2 tf = new TimeFormat2(null, null, null);
						tf.setStartTime(s1);
						tf.setEndTime(s3);
						tf.setTimeExp(unit[i].Time_Expression + "到" + unit[i + 2].Time_Expression);
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
						tf.setTimeExp(unit[i + 1].Time_Expression);
						ltf.add(tf);
						tf = new TimeFormat2(null, null, null);
						tf.setStartTime(s3);
						tf.setEndTime(s3);
						tf.setTimeExp(unit[i + 2].Time_Expression);
						ltf.add(tf);
					}
					i = i + 2;
				} else {
					TimeFormat2 tf = new TimeFormat2(null, null, null);
					String s5 = DateUtil.formatDateDefault(unit[i].getTime()) + " " + unit[i].getIsAllDayTime();
					// System.out.println("s5:" + unit[i].Time_Expression+s5);
					tf.setStartTime(s5);
					tf.setEndTime(s5);
					tf.setTimeExp(unit[i].Time_Expression);
					ltf.add(tf);
				}
			}
		} else {
			for (int i = 0; i < unit.length; i++) {
				String s4 = DateUtil.formatDateDefault(unit[i].getTime()) + " " + unit[i].getIsAllDayTime();
				// System.out.println("s4:" + unit[i].Time_Expression+s4);
				TimeFormat2 tf = new TimeFormat2(null, null, null);
				tf.setStartTime(s4);
				tf.setEndTime(s4);
				tf.setTimeExp(unit[i].Time_Expression);
				ltf.add(tf);
			}

		}

		return ltf;
	}

	public static String o_wordTranslator(String text, String exp) {
		if (text.contains("礼拜") && exp.contains("星期")) {
			exp = exp.replaceAll("星期", "礼拜");
		}
		if (text.contains("礼拜天") && exp.contains("礼拜7")) {
			exp = exp.replaceAll("礼拜7", "礼拜天");
		}
		if (text.contains("礼拜日") && exp.contains("礼拜7")) {
			exp = exp.replaceAll("礼拜7", "礼拜日");
		}
		if (text.contains("星期天") && exp.contains("星期7")) {
			exp = exp.replaceAll("星期7", "星期天");
		}
		if (text.contains("星期日") && exp.contains("星期7")) {
			exp = exp.replaceAll("星期7", "星期日");
		}
		if (text.contains("周天") && exp.contains("周7")) {
			exp = exp.replaceAll("周7", "周天");
		}
		if (text.contains("周日") && exp.contains("周7")) {
			exp = exp.replaceAll("周7", "周日");
		}
		if (text.contains("周末") && exp.contains("周7")) {
			exp = exp.replaceAll("周7", "周末");
		}

		String rules = "最近几天|未来几天|随后几天|未来3天|最近3天|这3天|未来三天|最近三天|这三天|这几天|这些天|往后三天|往后3天|接下来三天";
		Pattern p = Pattern.compile(rules);
		Matcher m = p.matcher(text);
		StringBuffer sb = new StringBuffer();
		boolean result = m.find();
		while (result) {
			if (exp.contains("明天到大后天")) {
				exp = exp.replaceAll("明天到大后天", m.group());
			}
			// m.appendReplacement(sb, "明天到大后天");
			result = m.find();
		}
		// m.appendTail(sb);
		// text = sb.toString();

		rules = "今明天|今天明天|今明天|今明两天|今明2天";
		p = Pattern.compile(rules);
		m = p.matcher(text);
		sb = new StringBuffer();
		result = m.find();
		while (result) {
			if (exp.contains("今天到明天")) {
				exp = exp.replaceAll("今天到明天", m.group());
			}
			// m.appendReplacement(sb, "明天到大后天");
			result = m.find();
		}
		rules = "今儿";
		p = Pattern.compile(rules);
		m = p.matcher(text);
		sb = new StringBuffer();
		result = m.find();
		while (result) {
			if (exp.contains("今天")) {
				exp = exp.replaceAll("今天", m.group());
			}
			// m.appendReplacement(sb, "明天到大后天");
			result = m.find();
		}
		
		
		rules = "明后天|明天后天|未来2天|最近2天|未来两天|最近两天|明后两天|明后2天|接下来两天";
		p = Pattern.compile(rules);
		m = p.matcher(text);
		sb = new StringBuffer();
		result = m.find();
		while (result) {
			if (exp.contains("明天到后天")) {
				exp = exp.replaceAll("明天到后天", m.group());
			}
			// m.appendReplacement(sb, "明天到大后天");
			result = m.find();
		}
		
		rules = "(这周|这星期|这一周|这一星期|最近一周|这1周|这1星期|最近1周)";
		p = Pattern.compile(rules);
		m = p.matcher(text);
		sb = new StringBuffer();
		result = m.find();
		while (result) {
			if (exp.contains("周1到周7")) {
				exp = exp.replaceAll("周1到周7", m.group());
			}
			// m.appendReplacement(sb, "明天到大后天");
			result = m.find();
		}
		
		rules = "圣诞节|平安夜|父亲节|元旦|除夕|春节|清明节|劳动节|端午节|中秋节|国庆节|母亲节|儿童节|建军节|愚人节|青年节圣诞节|平安夜|"
				+ "教师节|万圣节|植树节|重阳节|腊八节|情人节|元宵节|感恩节|妇女节|小年|五一|六一|七一|八一|九一|十一";
		p = Pattern.compile(rules);
		m = p.matcher(text);
		sb = new StringBuffer();
		result = m.find();
		while (result) {
			if (exp.contains("年")&&exp.contains("月")&&exp.contains("日")) {
//				exp = exp.replaceAll("年", m.group());
				exp = m.group();
			}
			// m.appendReplacement(sb, "明天到大后天");
			result = m.find();
		}

		return exp;
	}

	public static List<TimeFormat2> normalTime_o(String text, List<TimeFormat2> ltf) {
		for (TimeFormat2 t : ltf) {
//			System.out.println(t.getTimeExp());
//			System.out.println("text:" + text);
			String tar = stringPreHandlingModule.numberTranslator(text);
//			System.out.println("tar:" + tar);
			String exp = t.getTimeExp();
//			System.out.println("exp:" + exp);
			if (!text.contains(exp)) {
//				System.out.println("exp:" + exp);
				exp = o_wordTranslator(text, exp);
//				System.out.println("exp:" + exp);
				String s = foematIntegerUtil.regexTr(exp);
//				System.out.println("exps:" + s);
				String firststr = s.substring(0, 1);
				String endtstr = s.substring(s.length() - 1);
//				s = text.substring(text.indexOf(firststr), text.indexOf(firststr) + s.length());
				t.setTimeExp(s);
			}
		}

		return ltf;
	}
	public static List<TimeFormat3> normalTime_o2(String text, List<TimeFormat2> ltf) {
		List<TimeFormat3> ltf3 = new ArrayList<TimeFormat3>();
		String h1="null";String h2="null";
		if(ltf.size()>1){
//			 h1=ltf.get(0).getTimeExp()+ltf.get(1).getTimeExp();
			 h2=ltf.get(0).getTimeExp()+"和"+ltf.get(1).getTimeExp();
		}
//		if(text.contains(h1)){
//			TimeFormat3 tf3=new TimeFormat3(null, null);
//			List<String> ls=new ArrayList<String>();
//			ls.add(ltf.get(0).toString2());
//			ls.add(ltf.get(1).toString2());
//			tf3.setTime(ls);
//			tf3.setTimeExp(h1);
//			ltf3.add(tf3);	
//		}else
			if(text.contains(h2)){
			TimeFormat3 tf3=new TimeFormat3(null, null);
			List<String> ls=new ArrayList<String>();
			ls.add(ltf.get(0).toString2());
			ls.add(ltf.get(1).toString2());
			tf3.setTime(ls);
			tf3.setTimeExp(h2);
			ltf3.add(tf3);	
		}else {
			for(TimeFormat2 tf2:ltf){
				TimeFormat3 tf3=new TimeFormat3(null, null);
				List<String> ls=new ArrayList<String>();
				ls.add(tf2.toString2());
				tf3.setTime(ls);
				tf3.setTimeExp(tf2.getTimeExp());
				ltf3.add(tf3);
			}
		}
		
		
		return ltf3;
		 
		
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		// "今天到二十号天气本周日天气"; 礼拜四天气 "十二月九日天气";"周四";"十月十号";

		String text = "今天明天点天气";
		List<TimeFormat2> ltf = normalTime(text);
		List<TimeFormat2> ltf2 = normalTime_o(text, ltf);
		System.out.println("ltf.size():" + ltf.size());
		for (TimeFormat2 t : ltf2) {
			System.out.println(t.toString());
		}
		List<TimeFormat3> ltf3=normalTime_o2(text,ltf2);
		System.out.println("ltf3.size():" + ltf3.size());
		for (TimeFormat3 t : ltf3) {
			System.out.println(t.toString());
		}

	}

}
