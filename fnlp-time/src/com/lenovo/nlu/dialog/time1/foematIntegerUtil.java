package com.lenovo.nlu.dialog.time1;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class foematIntegerUtil {
//	static String[] units = { "", "十", "百", "千", "万", "十万", "百万", "千万", "亿", "十亿", "百亿", "千亿", "万亿" };
	static String[] units = { "", "十", "", "", "万", "十万", "百万", "千万", "亿", "十亿", "百亿", "千亿", "万亿" };
	static char[] numArray = { '零', '一', '二', '三', '四', '五', '六', '七', '八', '九' };

	private static String foematInteger(String num) {
		// char[] val = String.valueOf(num).toCharArray();
		char[] val = num.toCharArray();
		int len = val.length;
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < len; i++) {
			String m = val[i] + "";
 
				int n = Integer.valueOf(m);
//				System.out.println("n:" + n);
				boolean isZero = n == 0;
				String unit = units[(len - 1) - i];
//				System.out.println("unit:"+unit);
				if (isZero) {
					if ('0' == val[i - 1]) {
						// not need process if the last digital bits is 0
						continue;
					} else {
						// no unit for 0
						sb.append(numArray[n]);//System.out.println("numArray :"+numArray[n]);
					}
				} else {
					sb.append(numArray[n]);//System.out.println("numArray[n]:"+numArray[n]);
					sb.append(unit);
//					System.out.println("unit:"+unit);
				}
		 
		}
		return sb.toString();
	}

	static String regexTr(String num) {

		Pattern p = Pattern.compile("([0-9]{1,4})");//([0-9]{2,4})
		Matcher ma = p.matcher(num);
		StringBuffer sbs = new StringBuffer();
		boolean result = ma.find();
		while (result) {
		String group = ma.group();
//		System.out.println("group:" + group);
		String numStr2 = foematInteger(group);
//		System.out.println("numStr2:" + numStr2);
		if(num.contains(group+"年")){
			int i=Integer.parseInt(group);
			  numStr2=DateUtils.numToUpper(i);
		}
		if(num.contains(group+"月")){
			int i=Integer.parseInt(group);
			  numStr2=DateUtils.monthToUppder(i);
		}
		if(num.contains(group+"日")||num.contains(group+"号")){
			int i=Integer.parseInt(group);
			  numStr2=DateUtils.dayToUppder(i);
		}
//		System.out.println("numStr2:" + numStr2);
		ma.appendReplacement(sbs, numStr2); 
		result = ma.find();
		}
		ma.appendTail(sbs);
		String re =sbs.toString();
		return re;

		}

		public static void main(String[] args) {
		String num = " 12号天气";//"2016年10月20号天气"
//		String numStr = foematInteger(num);
		String s= regexTr(num);
		System.out.println("num= " + num + ", convert result: " + s);
		
//		String num2 = "11";
//		String numStr2 = foematInteger(num2);
//		if("一十零".equals(numStr2)){
//			numStr2="十";
//		}
//		if(numStr2.substring(numStr2.length()-1).equals("零")){
//			numStr2=numStr2.substring(0,numStr2.length()-1);
//		}
//		System.out.println("num2= " + num2 + ", convert result2: " + numStr2);
		}
		 
		
		
		
}
