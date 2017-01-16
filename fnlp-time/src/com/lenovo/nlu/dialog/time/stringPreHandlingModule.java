package com.lenovo.nlu.dialog.time;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.lenovo.nlu.dialog.time.holiday.StandardTime;
import com.lenovo.nlu.dialog.time.holiday.TimeFormat;

/**
 * 字符串预处理模块，为分析器TimeNormalizer提供相应的字符串预处理服务
 * 
 * @author 曹零07300720158
 *
 */
public class stringPreHandlingModule {

	/**
	 * 该方法删除一字符串中所有匹配某一规则字串
	 * 可用于清理一个字符串中的空白符和语气助词
	 * 
	 * @param target 待处理字符串
	 * @param rules 删除规则
	 * @return 清理工作完成后的字符串
	 */
	public static String delKeyword(String target, String rules){
		Pattern p = Pattern.compile(rules); 
		Matcher m = p.matcher(target); 
		StringBuffer sb = new StringBuffer(); 
		boolean result = m.find(); 
		while(result) { 
			m.appendReplacement(sb, ""); 
			result = m.find(); 
		}
		m.appendTail(sb);
		String s = sb.toString();
//		System.out.println("字符串："+target+" 的处理后字符串为：" +sb);
		return s;
	}
	
	public static String wordTranslator(String target){
		String rules ="今明天|今天明天|今明天|今明两天|今明2天";
		Pattern p = Pattern.compile(rules); 
		Matcher m = p.matcher(target); 
		StringBuffer sb = new StringBuffer(); 
		boolean result = m.find(); 
		while(result) { 
			m.appendReplacement(sb, "今天和明天"); 
			result = m.find(); 
		}
		m.appendTail(sb);
		String s = sb.toString();
//		System.out.println("字符串："+target+" 的处理后字符串为：" +sb);
		
		rules ="今儿";
		p = Pattern.compile(rules); 
		m = p.matcher(s); 
		sb = new StringBuffer(); 
		result = m.find(); 
		while(result) { 
			m.appendReplacement(sb, "今天"); 
			result = m.find(); 
		}
		m.appendTail(sb);
		s = sb.toString();
//		System.out.println("字符串："+target+" 的处理后字符串为：" +sb);
		
		rules ="明后天|明天后天|未来2天|最近2天|未来两天|最近两天|明后两天|明后2天";
		p = Pattern.compile(rules); 
		m = p.matcher(s); 
		sb = new StringBuffer(); 
		result = m.find(); 
		while(result) { 
			m.appendReplacement(sb, "明天和后天"); 
			result = m.find(); 
		}
		m.appendTail(sb);
		s = sb.toString();
//		System.out.println("字符串："+target+" 的处理后字符串为：" +sb);
		
		rules ="最近几天|未来几天|随后几天|未来3天|最近3天|这3天|未来三天|最近三天|这三天|这几天|这些天|往后三天|往后3天";
		p = Pattern.compile(rules); 
		m = p.matcher(s); 
		sb = new StringBuffer(); 
		result = m.find(); 
		while(result) { 
			m.appendReplacement(sb, "明天和后天和大后天"); 
			result = m.find(); 
		}
		m.appendTail(sb);
		s = sb.toString();
//		System.out.println("字符串："+target+" 的处理后字符串为：" +sb);
		
		rules ="礼拜";
		p = Pattern.compile(rules); 
		m = p.matcher(s); 
		sb = new StringBuffer(); 
		result = m.find(); 
		while(result) { 
			m.appendReplacement(sb, "星期"); 
			result = m.find(); 
		}
		m.appendTail(sb);
		s = sb.toString();
//		System.out.println("字符串："+target+" 的处理后字符串为：" +sb);
		
		rules ="这周|这星期|这一周|这一星期|本周|本星期|最近一周|这1周|这1星期|最近1周";
		p = Pattern.compile(rules); 
		m = p.matcher(s); 
		sb = new StringBuffer(); 
		result = m.find(); 
		while(result) { 
			m.appendReplacement(sb, "周1到周7"); 
			result = m.find(); 
		}
		m.appendTail(sb);
		s = sb.toString();
//		System.out.println("字符串："+target+" 的处理后字符串为：" +sb);
		
		rules ="[一二两三四五六七八九十0123456789][号日][一二两三四五六七八九十0123456789][号日]";
		p = Pattern.compile(rules); 
		m = p.matcher(s); 
		
		sb = new StringBuffer(); 
		result = m.find(); 
		while(result) { 
//			System.out.println("m.group(0):"+m.group(0));
			String rules2="号|日";
			Pattern p2=p = Pattern.compile(rules2); 
			Matcher m2 = p.matcher(m.group(0));
			StringBuffer sb2 = new StringBuffer(); 
			boolean result2 = m2.find(); 
			while(result2) { 
//				System.out.println("m2.group(0):"+m2.group(0));
				m2.appendReplacement(sb2, "号和"); 
				result2 = m2.find();
			}
			m2.appendTail(sb2);
			String s2=sb2.toString();
			m.appendReplacement(sb, s2);
			result = m.find(); 
		}
//		System.out.println("sb:"+sb);
		m.appendTail(sb);
		s = sb.toString();
//		System.out.println("字符串："+target+" 的处理后字符串为：" +sb);
		
		rules ="(周|星期)[1234567日天](周|星期)[1234567日天]";
		p = Pattern.compile(rules); 
		m = p.matcher(s); 
		
		sb = new StringBuffer(); 
		result = m.find(); 
		while(result) { 
//			System.out.println("m.group(0):"+m.group(0));
			String rules2="周|星期";
			Pattern p2=p = Pattern.compile(rules2); 
			Matcher m2 = p.matcher(m.group(0));
			StringBuffer sb2 = new StringBuffer(); 
			boolean result2 = m2.find(); 
			while(result2) { 
//				System.out.println("m2.group(0):"+m2.group(0));
				m2.appendReplacement(sb2, "和周"); 
				result2 = m2.find();
			}
			m2.appendTail(sb2);
			String s2=sb2.toString();
			m.appendReplacement(sb, s2);
			result = m.find(); 
		}
//		System.out.println("sb:"+sb);
		m.appendTail(sb);
		s = sb.toString();
//		System.out.println("字符串："+target+" 的处理后字符串为：" +sb);

		rules ="圣诞节|平安夜|父亲节|元旦|除夕|春节|清明节|劳动节|端午节|中秋节|国庆节|母亲节|儿童节|建军节|愚人节|青年节圣诞节|平安夜|"
				+ "教师节|万圣节|植树节|重阳节|腊八节|情人节|元宵节|感恩节|妇女节|小年|五一|六一|七一|八一|九一|十一";
		p = Pattern.compile(rules); 
		m = p.matcher(s); 
		sb = new StringBuffer(); 
		result = m.find(); 
		StandardTime standardTime = new StandardTime();
		
		while(result) { 	
			TimeFormat tfFormat = standardTime.normTime(m.group());
			m.appendReplacement(sb, tfFormat.getStartTime()); 
			result = m.find(); 
		}
		m.appendTail(sb);
		s = sb.toString();
//		System.out.println("字符串："+target+" 的处理后字符串为：" +sb);

		
		////////////////////////////////////////////
		rules ="最近";
		p = Pattern.compile(rules); 
		m = p.matcher(s); 
		sb = new StringBuffer(); 
		result = m.find(); 
		while(result) { 
			m.appendReplacement(sb, "明天和后天"); 
			result = m.find(); 
		}
		m.appendTail(sb);
		s = sb.toString();
//		System.out.println("字符串："+target+" 的处理后字符串为：" +sb);
		
		return s;
		
	}
	
	/**
	 * 该方法可以将字符串中所有的用汉字表示的数字转化为用阿拉伯数字表示的数字
	 * 如"这里有一千两百个人，六百零五个来自中国"可以转化为
	 * "这里有1200个人，605个来自中国"
	 * 此外添加支持了部分不规则表达方法
	 * 如两万零六百五可转化为20650
	 * 两百一十四和两百十四都可以转化为214
	 * 一六零加一五八可以转化为160+158
	 * 该方法目前支持的正确转化范围是0-99999999
	 * 该功能模块具有良好的复用性
	 * 
	 * @param target 待转化的字符串
	 * @return 转化完毕后的字符串
	 */
	public static String numberTranslator(String target){
		Pattern p = Pattern.compile("[一二两三四五六七八九123456789]万[一二两三四五六七八九123456789](?!(千|百|十))"); 
		Matcher m = p.matcher(target); 
		StringBuffer sb = new StringBuffer(); 
		boolean result = m.find(); 
		while(result) { 
			String group = m.group();
			String[] s = group.split("万");
			int num = 0;
			if(s.length == 2){
				num += wordToNumber(s[0])*10000 + wordToNumber(s[1])*1000;
			}
			m.appendReplacement(sb, Integer.toString(num)); 
			result = m.find(); 
		}
		m.appendTail(sb);
		target = sb.toString();
		
		p = Pattern.compile("[一二两三四五六七八九123456789]千[一二两三四五六七八九123456789](?!(百|十))"); 
		m = p.matcher(target); 
		sb = new StringBuffer(); 
		result = m.find(); 
		while(result) { 
			String group = m.group();
			String[] s = group.split("千");
			int num = 0;
			if(s.length == 2){
				num += wordToNumber(s[0])*1000 + wordToNumber(s[1])*100;
			}
			m.appendReplacement(sb, Integer.toString(num)); 
			result = m.find(); 
		}
		m.appendTail(sb);
		target = sb.toString();
		
		p = Pattern.compile("[一二两三四五六七八九123456789]百[一二两三四五六七八九123456789](?!十)"); 
		m = p.matcher(target); 
		sb = new StringBuffer(); 
		result = m.find(); 
		while(result) { 
			String group = m.group();
			String[] s = group.split("百");
			int num = 0;
			if(s.length == 2){
				num += wordToNumber(s[0])*100 + wordToNumber(s[1])*10;
			}
			m.appendReplacement(sb, Integer.toString(num)); 
			result = m.find(); 
		}
		m.appendTail(sb);
		target = sb.toString();
		
		p = Pattern.compile("[零一二两三四五六七八九]"); 
		m = p.matcher(target); 
		sb = new StringBuffer(); 
		result = m.find(); 
		while(result) { 
			m.appendReplacement(sb, Integer.toString(wordToNumber(m.group()))); 
			result = m.find(); 
		}
		m.appendTail(sb);
		target = sb.toString();
		
		p = Pattern.compile("(?<=(周|星期|礼拜))[末天日]"); 
		m = p.matcher(target); 
		sb = new StringBuffer(); 
		result = m.find(); 
		while(result) { 
			m.appendReplacement(sb, Integer.toString(wordToNumber(m.group()))); 
			result = m.find(); 
		}
		m.appendTail(sb);
		target = sb.toString();
		
		p = Pattern.compile("(?<!(周|星期))0?[0-9]?十[0-9]?"); 
		m = p.matcher(target);
		sb = new StringBuffer();
		result = m.find();
		while(result) { 
			String group = m.group();
			String[] s = group.split("十");
			int num = 0;
			if(s.length == 0){
				num += 10;
			}
			else if(s.length == 1){
				int ten = Integer.parseInt(s[0]);
				if(ten == 0)
					num += 10;
				else num += ten*10;
			}
			else if(s.length == 2){
				if(s[0].equals(""))
					num += 10;
				else{
					int ten = Integer.parseInt(s[0]);
					if(ten == 0)
						num += 10;
					else num += ten*10;
				}
				num += Integer.parseInt(s[1]);
			}
			m.appendReplacement(sb, Integer.toString(num)); 
			result = m.find(); 
		}
		m.appendTail(sb);
		target = sb.toString();
		
		p = Pattern.compile("0?[1-9]百[0-9]?[0-9]?"); 
		m = p.matcher(target);
		sb = new StringBuffer();
		result = m.find();
		while(result) { 
			String group = m.group();
			String[] s = group.split("百");
			int num = 0;
			if(s.length == 1){
				int hundred = Integer.parseInt(s[0]);
				num += hundred*100;
			}
			else if(s.length == 2){
				int hundred = Integer.parseInt(s[0]);
				num += hundred*100;
				num += Integer.parseInt(s[1]);
			}
			m.appendReplacement(sb, Integer.toString(num)); 
			result = m.find(); 
		}
		m.appendTail(sb);
		target = sb.toString();
		
		p = Pattern.compile("0?[1-9]千[0-9]?[0-9]?[0-9]?"); 
		m = p.matcher(target);
		sb = new StringBuffer();
		result = m.find();
		while(result) { 
			String group = m.group();
			String[] s = group.split("千");
			int num = 0;
			if(s.length == 1){
				int thousand = Integer.parseInt(s[0]);
				num += thousand*1000;
			}
			else if(s.length == 2){
				int thousand = Integer.parseInt(s[0]);
				num += thousand*1000;
				num += Integer.parseInt(s[1]);
			}
			m.appendReplacement(sb, Integer.toString(num)); 
			result = m.find(); 
		}
		m.appendTail(sb);
		target = sb.toString();
		
		p = Pattern.compile("[0-9]+万[0-9]?[0-9]?[0-9]?[0-9]?"); 
		m = p.matcher(target);
		sb = new StringBuffer();
		result = m.find();
		while(result) { 
			String group = m.group();
			String[] s = group.split("万");
			int num = 0;
			if(s.length == 1){
				int tenthousand = Integer.parseInt(s[0]);
				num += tenthousand*10000;
			}
			else if(s.length == 2){
				int tenthousand = Integer.parseInt(s[0]);
				num += tenthousand*10000;
				num += Integer.parseInt(s[1]);
			}
			m.appendReplacement(sb, Integer.toString(num)); 
			result = m.find(); 
		}
		m.appendTail(sb);
		target = sb.toString();
		
		return target;
	}
	
	/**
	 * 方法numberTranslator的辅助方法，可将[零-九]正确翻译为[0-9]
	 * 
	 * @param s 大写数字
	 * @return 对应的整形数，如果不是大写数字返回-1
	 */
	private static int wordToNumber(String s){
		if(s.equals("零")||s.equals("0"))
			return 0;
		else if(s.equals("一")||s.equals("1"))
			return 1;
		else if(s.equals("二")||s.equals("两")||s.equals("2"))
			return 2;
		else if(s.equals("三")||s.equals("3"))
			return 3;
		else if(s.equals("四")||s.equals("4"))
			return 4;
		else if(s.equals("五")||s.equals("5"))
			return 5;
		else if(s.equals("六")||s.equals("6"))
			return 6;
		else if(s.equals("七")||s.equals("天")||s.equals("日") || s.equals("末") ||s.equals("7"))
			return 7;
		else if(s.equals("八")||s.equals("8"))
			return 8;
		else if(s.equals("九")||s.equals("9"))
			return 9;
		else return -1;
	}
}







