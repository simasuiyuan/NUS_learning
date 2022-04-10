import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.sql.*;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.api.java.UDF2;
import org.apache.spark.sql.api.java.UDF4;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.types.ArrayType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.graphframes.GraphFrame;
import org.graphframes.lib.AggregateMessages;
import org.graphframes.lib.Pregel;
import scala.Serializable;
import scala.collection.mutable.WrappedArray;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.apache.spark.sql.functions.*;

public class FindPath {
    // From: https://stackoverflow.com/questions/3694380/calculating-distance-between-two-points-using-latitude-longitude
    private static double distance(double lat1, double lat2, double lon1, double lon2) {
        final int R = 6371; // Radius of the earth
        double latDistance = Math.toRadians(lat2 - lat1);
        double lonDistance = Math.toRadians(lon2 - lon1);
        double a = Math.sin(latDistance / 2) * Math.sin(latDistance / 2)
                + Math.cos(Math.toRadians(lat1)) * Math.cos(Math.toRadians(lat2))
                * Math.sin(lonDistance / 2) * Math.sin(lonDistance / 2);
        double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        double distance = R * c * 1000; // convert to meters
        double height = 0; // For this assignment, we assume all locations have the same height.
        distance = Math.pow(distance, 2) + Math.pow(height, 2);
        return Math.sqrt(distance);
    }

    private static Dataset<Row> getNodeDF(SQLContext sqlContext, String raw_osm_file, String createTempViewName){
        Map<String, String> options = new HashMap<>();
        List<StructField> fields = new ArrayList<StructField>();
        fields.add(DataTypes.createStructField("_id", DataTypes.LongType, true));
        fields.add(DataTypes.createStructField("_lat", DataTypes.DoubleType, true));
        fields.add(DataTypes.createStructField("_lon", DataTypes.DoubleType, true));
        StructType schema = DataTypes.createStructType(fields);
        options.put("rowTag", "node");
        Dataset<Row> nodeDF = sqlContext.read()
                .options(options)
                .format("xml")
                .schema(schema)
                .load(raw_osm_file)
                .withColumnRenamed("_id", "node_id")
                .withColumnRenamed("_lat", "latitude")
                .withColumnRenamed("_lon", "longitude");
        if(!createTempViewName.isEmpty()) nodeDF.createOrReplaceTempView(createTempViewName);
        return nodeDF;
    }

    private static Dataset<Row> getRoadDF(SQLContext sqlContext, String raw_osm_file, String createTempViewName){
        Map<String, String> options = new HashMap<>();
        List<StructField> fields = new ArrayList<StructField>(); //reset fields for ways
        fields.add(DataTypes.createStructField("_id", DataTypes.LongType, true));
        List<StructField> inner_fields = new ArrayList<StructField>();
        inner_fields.add(DataTypes.createStructField("_ref", DataTypes.LongType, true));
        ArrayType nd_type = DataTypes.createArrayType(DataTypes.createStructType(inner_fields));
        fields.add(DataTypes.createStructField("nd", nd_type, true));
        inner_fields = new ArrayList<StructField>();
        inner_fields.add(DataTypes.createStructField("_k", DataTypes.StringType, true));
        inner_fields.add(DataTypes.createStructField("_v", DataTypes.StringType, true));
        ArrayType tag_type = DataTypes.createArrayType(DataTypes.createStructType(inner_fields));
        fields.add(DataTypes.createStructField("tag", tag_type, true));
        StructType schema = DataTypes.createStructType(fields);
        options.put("rowTag", "way");
        Dataset<Row> roadDF = sqlContext.read()
                .options(options)
                .format("xml")
                .schema(schema)
                .load(raw_osm_file)
                .withColumnRenamed("_id", "road_id");
        if(!createTempViewName.isEmpty()){
            roadDF.createOrReplaceTempView(createTempViewName);
        } else {
            roadDF.createOrReplaceTempView("roadDF");
        }
        roadDF = sqlContext.sql(
                "SELECT *, checkOneWay(tag) AS oneway " +
                        "FROM roadDF " +
                        "WHERE array_contains(tag._k, \"highway\")");
        if(!createTempViewName.isEmpty()) roadDF.createOrReplaceTempView(createTempViewName);
        return roadDF;
    }

    private static Dataset<Row> getRelationDF(SQLContext sqlContext, String nodeDFViewName, String roadDFViewName, String createTempViewName){

        Dataset<Row> expwayNodeDF = sqlContext.sql(
                String.format("SELECT road_id, exploded.nd_index+1 AS indexedNode, node._ref AS node_id, oneway " +
                        "FROM %s " +
                        "lateral view posexplode(%s.nd) exploded AS nd_index, node", roadDFViewName, roadDFViewName));
        expwayNodeDF.createOrReplaceTempView("expwayNodeDF");

        /* forward_relation: forward relation (bi-direction & oneway)*/
        Dataset<Row> forward_relation = sqlContext.sql(
                String.format("WITH " +
                        "sourceDF AS (" +
                        "SELECT road_id, indexedNode AS indexedNode_src, src_node.node_id AS source, node_info.latitude AS src_latitude, node_info.longitude AS src_longitude, oneway " +
                        "FROM expwayNodeDF AS src_node, %s AS node_info " +
                        "WHERE src_node.node_id == node_info.node_id), " +
                        "destDF AS (" +
                        "SELECT road_id, indexedNode - 1 AS indexedNode_dest, dest_node.node_id AS destination, node_info.latitude AS dest_latitude, node_info.longitude AS dest_longitude " +
                        "FROM expwayNodeDF AS dest_node, %s AS node_info " +
                        "WHERE dest_node.node_id == node_info.node_id) " +
                        "SELECT s.road_id AS road_id, s.source AS source, d.destination AS destination, " +
                        "(CASE WHEN d.destination IS NULL " +
                        "THEN NULL " +
                        "ELSE computeDistance(s.src_latitude, d.dest_latitude, s.src_longitude, d.dest_longitude) END) AS distance, s.oneway " +
                        "FROM sourceDF s LEFT JOIN destDF d " +
                        "ON s.road_id == d.road_id AND s.indexedNode_src == d.indexedNode_dest ", nodeDFViewName, nodeDFViewName));//computeDistance

        /* reversed_relation: reversed relation (oneway)*/
        Dataset<Row> reversed_relation = forward_relation.where("oneway = false")
                .withColumn("rev_src",forward_relation.col("destination"))
                .withColumn("rev_dest", forward_relation.col("source"))
                .drop("destination", "source")
                .withColumnRenamed("rev_src", "source")
                .withColumnRenamed("rev_dest", "destination")
                .select("road_id", "source", "destination", "distance", "oneway");
        Dataset<Row> full_relation = forward_relation.union(reversed_relation).where("source IS NOT NULL").repartition(col("source"));
        if(!createTempViewName.isEmpty()) full_relation.createOrReplaceTempView(createTempViewName);
        return full_relation;
    }

    private static Dataset<Row> getAdjMapDf(SQLContext sqlContext){
        return sqlContext.sql(
                "SELECT concat_ws(\" \", source, sort_array(collect_set(destination),true)) as output " +
                        "FROM relationDF " +
                        "GROUP BY source " +
                        "ORDER BY source");
    }

    private static String extractStringPath(SQLContext sqlContext, Dataset<Row> paths){
        String edge_vertices = new String();
        for(String col: paths.columns()){
            if(col.contains("v")) edge_vertices += String.format("%s.id, ", col);
        }
        paths.createOrReplaceTempView("paths");
        Row result = sqlContext.sql(
                "SELECT from.id, " +
                        edge_vertices + "to.id " +
                        "FROM paths"
        ).first();
        String res_string = new String();
        for(int i=0; i<result.size(); i++){
            if(i<result.size()-1){
                res_string+=String.format("%d -> ", result.getLong(i));
            } else {
                res_string+=Long.toString(result.getLong(i));
            }
        }
        return res_string;
    }

    private static String getPathsFromBFS(SQLContext sqlContext, GraphFrame g, String from_node, String to_node){
        from_node = String.format("id = %s", from_node);
        to_node = String.format("id = %s", to_node);
        g.cache();
        Dataset<Row> paths = g.bfs().fromExpr(from_node).toExpr(to_node).maxPathLength((int) (g.vertices().count() - 2)).run();
        paths.show();
        return extractStringPath(sqlContext, paths);
    }

    private static class Task{
        private String src;
        private String dst;
        Task() {
            setSrc(null);
            setDst(null);
        }
        Task(String src, String dst){
            setSrc(src);
            setDst(dst);
        }
        public String getDst() {
            return dst;
        }
        public String getSrc() {
            return src;
        }
        public void setSrc(String src){
            this.src=src;
        }
        public void setDst(String dst) {
            this.dst = dst;
        }
    }

    private static String getShortestPathsFromDijkstraV2(SQLContext sqlContext, String from_node, String to_node, long noOfNodes){
        if (sqlContext.sql(String.format("SELECT count(*) as count FROM nodeDF where nodeDF.node_id = %s", to_node)).first().getLong(0)==0){
            return null;
        }

        Dataset<Row> candidate_paths = sqlContext.sql(
                String.format(
                        "SELECT source AS src, destination AS latest_node, distance AS total_distance, " +
                                "addPath(CAST(source AS STRING), destination) AS path, " +
                                "array(source, destination) AS visited " +
                        "FROM relationDF WHERE (source=%s) AND (distance IS NOT NULL)", from_node));

        candidate_paths.createOrReplaceTempView("candidatePaths");

        long MAX_PATH = noOfNodes - 2;
        for(int i=1; i < MAX_PATH; i++) {
            if(!candidate_paths.where(String.format("latest_node = %s", to_node)).isEmpty()){
                return candidate_paths.where(String.format("latest_node = %s", to_node)).select("path").first().getString(0);
            }
            candidate_paths = sqlContext.sql(
                    String.format(
                            "SELECT C.src AS src, R.destination AS latest_node, " +
                                    "(C.total_distance + R.distance) AS total_distance, " +
                                    "array_union(C.visited, array(R.destination)) AS visited, " +
                                    "addPath(C.path, R.destination) AS path " +
                                    "FROM candidatePaths C " +
                                    "LEFT JOIN relationDF R ON C.latest_node = R.source " +
                                    "WHERE ((R.distance IS NOT NULL) AND (NOT array_contains(C.visited, R.destination)))"));
            candidate_paths = candidate_paths.sort("total_distance");
            candidate_paths.createOrReplaceTempView("candidatePaths");
        }
        return null;
    }

    // this is able to solve 5 cases provided, but not all extra
    /*unsolvable cases:
    5239881814 6977240310
    1345143432 1854366392
    1345143432 1832488252
    1534829927 1832488252
    5936493264 1345143432
    */
    private static String getShortestPathsFromDijkstra(SQLContext sqlContext, GraphFrame g, String from_node, String to_node){
        if (sqlContext.sql(String.format("SELECT count(*) as count FROM nodeDF where nodeDF.node_id = %s", to_node)).first().getLong(0)==0){
            return null;
        }

        Dataset<Row> vertices_cached = AggregateMessages.getCachedDataFrame(
                g.vertices().withColumn("visited", lit(false))
                        .withColumn("total_distance", when(g.vertices().col("id").equalTo(from_node), 0).otherwise(Double.POSITIVE_INFINITY))
                        .withColumn("path", lit("")));
        Dataset<Row> edges_cached = AggregateMessages.getCachedDataFrame(g.edges().where(g.edges().col("distance").isNotNull()));
        GraphFrame g_Dijkstra = GraphFrame.apply(vertices_cached, edges_cached);

        long MAX_PATH = g_Dijkstra.vertices().count() - 1;
        for(int i=1; i < MAX_PATH; i++){//
            Row current_node = g_Dijkstra.vertices().where("visited = false").sort("total_distance").first();
            Long current_node_id = current_node.getLong(current_node.fieldIndex("id"));

            Column msg_distance = Pregel.src("total_distance").plus(Pregel.edge("distance"));

            Column msg_path = when(Pregel.src("id").cast(DataTypes.StringType).equalTo(from_node), Pregel.src("id").cast(DataTypes.StringType))
                    .otherwise(callUDF("addPath", Pregel.src("path"), Pregel.src("id")));

            Column msg_for_dst = when(Pregel.src("id").equalTo(current_node_id), struct(msg_distance, msg_path));

            Dataset<Row> new_distances = g_Dijkstra.aggregateMessages().sendToDst(msg_for_dst).agg(min(AggregateMessages.msg()).alias("aggMsg"));

            Column new_visited_col = when(g_Dijkstra.vertices().col("visited")
                            .or(g_Dijkstra.vertices().col("id").equalTo(current_node_id)),
                    true).otherwise(false);

            Column new_distance_col = when(new_distances.col("aggMsg").isNotNull()
                            .and(new_distances.col("aggMsg").getItem("col1").$less(g_Dijkstra.vertices().col("total_distance"))),
                    new_distances.col("aggMsg").getItem("col1")).otherwise(g_Dijkstra.vertices().col("total_distance"));
            Column new_path_col = when(new_distances.col("aggMsg").isNotNull()
                            .and(new_distances.col("aggMsg").getItem("col1").$less(Double.POSITIVE_INFINITY)),
                    new_distances.col("aggMsg").getItem("col2").cast("string")).otherwise(g_Dijkstra.vertices().col("path"));

            Dataset<Row> combined_vertices = g_Dijkstra.vertices().join(new_distances,
                    g_Dijkstra.vertices().col("id").equalTo(new_distances.col("id")), "left_outer").drop(new_distances.col("id"));

            Dataset<Row> new_vertices = combined_vertices
                    .withColumn("visited", new_visited_col)
                    .withColumn("new_total_distance",new_distance_col)
                    .withColumn("newPath",new_path_col)
                    .drop("aggMsg", "total_distance", "path", "latitude", "longitude")
                    .withColumnRenamed("new_total_distance", "total_distance")
                    .withColumnRenamed("newPath", "path");

            Dataset<Row> new_vertices_cached = AggregateMessages.getCachedDataFrame(new_vertices);
            g_Dijkstra = GraphFrame.apply(new_vertices_cached, g_Dijkstra.edges());

            Row top_vertex = g_Dijkstra.vertices().where(g_Dijkstra.vertices().col("id").equalTo(to_node)).first();

            if(top_vertex.getBoolean(top_vertex.fieldIndex("visited"))){
                Dataset<Row> resultDF = g_Dijkstra.vertices().where(g_Dijkstra.vertices().col("id").equalTo(to_node));
                resultDF = resultDF.withColumn("newPath", callUDF("addPath",resultDF.col("path"), resultDF.col("id")))
                        .drop("visited", "path")
                        .withColumnRenamed("newPath", "path");
                g_Dijkstra.vertices().unpersist();
                return resultDF.select("path").first().getString(0);
            }
        }
        return null;
    }

    private static List<String> getShortestPathsFromDijkstraV3(SQLContext sqlContext, long noOfNodes, RunTimeReport run_time_report){
        sqlContext.cacheTable("relationDF");
        Dataset<Row> validTaskDF = sqlContext.sql(
                "SELECT taskDF.taskId, taskDF.from_node, taskDF.to_node " +
                        "FROM taskDF " +
                        "where taskDF.to_node IN (SELECT node_id FROM nodeDF)");
        validTaskDF.createOrReplaceTempView("taskDF");

        Dataset<Row> candidate_paths = sqlContext.sql(
                String.format(
                        "SELECT T.taskId," +
                                "R.source AS src, T.to_node, " +
                                "R.destination AS latest_node, R.distance AS total_distance, " +
                                "array(R.source, R.destination) AS visited, " +
                                "addPath(CAST(R.source AS STRING), R.destination) AS path, " +
                                "false AS found " +
                                "FROM relationDF R, taskDF T WHERE (R.source=T.from_node) AND (R.distance IS NOT NULL)"));
        Dataset<Row> resultDF = sqlContext.createDataFrame(new ArrayList<Row>(), candidate_paths.schema());
        resultDF.createOrReplaceTempView("resultDF");
        candidate_paths.createOrReplaceTempView("candidatePaths");
//
        long MIN_CHECK = 0;
        long MAX_CANDIDATES = 2;
        long MAX_PATH = 100;//20 mins = 1200 sec => 1200 sec / 100 = 12 / ;20 - 5 local - 1hr
        for(int i=1; i < noOfNodes-2; i++) {
//            candidate_paths = candidate_paths.withColumn("found", when(col("to_node").equalTo(col("latest_node")), true).otherwise(false));
            if(candidate_paths.where("found=true").limit(1).count()>0){
//                path_found.createOrReplaceTempView("pathFoundDF");
                resultDF = resultDF.union(candidate_paths.where("found=true"));
                resultDF.createOrReplaceTempView("resultDF");
//                candidate_paths = sqlContext.sql(
//                        String.format(
//                                "SELECT * FROM candidatePaths " +
//                                        "WHERE taskId NOT IN (" +
//                                        "SELECT taskId FROM resultDF GROUP BY taskId HAVING count(*) >= %d)", MAX_CANDIDATES
//                        )
//                );
//                candidate_paths.createOrReplaceTempView("candidatePaths");
            }
            long IsJobLeft = sqlContext.sql(
                    "SELECT * FROM taskDF WHERE taskId NOT IN (SELECT taskId FROM resultDF) LIMIT 1"
            ).count();
            if(((i > MIN_CHECK) & (IsJobLeft==0)) | (i > MAX_PATH)){
                run_time_report.addInfo(String.format("Number of iteration = %d", i), false);
                break;
            }
            candidate_paths = sqlContext.sql(
                    String.format(
                            "SELECT /*+ BROADCASTJOIN(relationDF) */C.taskId AS taskId, C.src AS src, C.to_node AS to_node, " +
                                    "R.destination AS latest_node, " +
                                    "(C.total_distance + R.distance) AS total_distance, " +
                                    "array_union(C.visited, array(R.destination)) AS visited, " +
                                    "addPath(C.path, R.destination) AS path, " +
                                    "CASE WHEN C.to_node=R.destination THEN true ELSE false END AS found " +
                                    "FROM candidatePaths C " +
                                    "LEFT JOIN relationDF R ON C.latest_node = R.source " +
                                    "WHERE ((C.found=false) AND (R.distance IS NOT NULL) AND (NOT array_contains(C.visited, R.destination)))")).repartition(col("latest_node"));
            candidate_paths.createOrReplaceTempView("candidatePaths");
        }

        List<String> all_results = resultDF
                .withColumn("min_distance_row", row_number().over(Window.partitionBy("taskId").orderBy("total_distance")))
                .where("min_distance_row = 1")
                .select("path")
                .map(new MapFunction<Row, String>() {
                    @Override
                    public String call(Row row) throws Exception {
                        return row.getString(0);
                    }
                }, Encoders.STRING()).collectAsList();
        return all_results;
    }

    private static class RunTimeReport implements Serializable {
        private static final String BOUNDARY_LINE = "========================================\n";
        private String Report = "\n"+BOUNDARY_LINE+"CS5424 Assignment 2 Runtime report\n"+BOUNDARY_LINE;
        private long start;

        RunTimeReport(long start){
            this.start = start;
        }
        private Float getRunTime(){
            long end = System.currentTimeMillis();
            return (end - this.start) / 1000F;
        }
        private void addInfo(String info, Boolean milestone){
            this.Report += info + "\n";
            if(milestone) {
                this.Report +=String.format("Runtime: %f seconds\n", getRunTime());
                this.Report += BOUNDARY_LINE;
            }
        }
        private String getReport(){
            return this.Report;
        }
        private void printReport(){
            System.out.println(this.Report);
        }
    }

    /* Define & register UDFS */
    private static void register_UDFs(SQLContext sqlContext){
        sqlContext.udf().register("checkOneWay", new UDF1<WrappedArray<Row>, Boolean>(){
            private static final long serialVersionUID = -5372447039252716846L;
            @Override
            public Boolean call(WrappedArray<Row> tags) throws Exception {
                for(int i=0; i<tags.size(); i++){
                    Row tag = tags.apply(i);
                    if(tag.getString(0).equals("oneway") & tag.getString(1).equals("yes")){
                        return true;
                    }
                }
                return false;
            }
        }, DataTypes.BooleanType);

        sqlContext.udf().register("computeDistance", new UDF4<Double, Double, Double, Double, Double>(){
            private static final long serialVersionUID = -5372447039252716846L;
            @Override
            public Double call(Double lat1, Double lat2, Double lon1, Double lon2) throws Exception {
                return distance(lat1, lat2, lon1, lon2);
            }
        }, DataTypes.DoubleType);

        sqlContext.udf().register("addPath", new UDF2<String, Long, String>(){
            @Override
            public String call(String path, Long id) throws Exception {
                return String.format("%s -> %d", path, id);
            }
        }, DataTypes.StringType);
    }

    /*Define global variables */
    static boolean runOnCluster = true;

    public static void main(String[] args) throws IOException {
        SparkConf sparkConf = new SparkConf().setAppName("FindPath");
        if (!runOnCluster) {
            sparkConf.setMaster("local[5]");
        } else {
            sparkConf.setMaster("local[10]").set("spark.executor.memory" ,"10g").set("spark.driver.memory", "12g");;
        }
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);
        SQLContext sqlContext = new SQLContext(jsc);
        String raw_osm_file = args[0];
        String input_src_dst_file = args[1];
        String result_adjmap_file = args[2];
        String result_path_file = args[3];
        FileSystem fs = FileSystem.get(jsc.hadoopConfiguration());
        String file = null;

        new GraphFrame();
        RunTimeReport run_time_report = new RunTimeReport(System.currentTimeMillis());

        register_UDFs(sqlContext);

        /* construct relation table */
        Dataset<Row> nodeDF = getNodeDF(sqlContext, raw_osm_file, "nodeDF");
        Dataset<Row> roadDF = getRoadDF(sqlContext, raw_osm_file, "roadDF");
        Dataset<Row> relationDF = getRelationDF(sqlContext, "nodeDF", "roadDF","relationDF");

//        Dataset<Row> adjMapCollection = getAdjMapDf(sqlContext);
//        run_time_report.addInfo("adjacency map constructed", true);
//
//        /* output adjacency map to hdfs */
//        adjMapCollection.select("output").coalesce(1).write().text("output");
//        file = fs.globStatus(new Path(String.format("%s/part*", "output")))[0].getPath().getName();
//        System.out.println(file);
//        fs.rename(new Path(String.format("%s/%s", "output",file)), new Path(result_adjmap_file));


        GraphFrame g = GraphFrame
                .apply(nodeDF.withColumnRenamed("node_id", "id"),
                        relationDF.withColumnRenamed("source", "src").withColumnRenamed("destination", "dst"));


        String ALGORITHM = "DijkstraV3";
        long noOfNodes = nodeDF.count();
        List<String> all_results = new ArrayList<String>();
        BufferedReader input_tasks=new BufferedReader(new InputStreamReader(fs.open(new Path(input_src_dst_file))));
        String task_line = input_tasks.readLine();

        switch (ALGORITHM) {
            case "BFS":
                /*BFS*/
                sqlContext.cacheTable("relationDF");
                while(task_line != null) {
                    String[] task = task_line.split(" ");
                    run_time_report.addInfo(String.format("BFS (no-weight) Processing task: \nsrc node: %s -> dst node: %s", task[0], task[1]), false);
                    String BFS_result = getPathsFromBFS(sqlContext, g, task[0], task[1]);
                    //check run time for task
                    run_time_report.addInfo(String.format("path found: %s", BFS_result), true);
                    all_results.add(BFS_result);
                    task_line = input_tasks.readLine();
                }
                break;
            case "Dijkstra":
                /*Dijkstra*/
                sqlContext.cacheTable("relationDF");
                while(task_line != null) {
                    String[] task = task_line.split(" ");
                    run_time_report.addInfo(String.format("Dijkstra Processing task: \nsrc node: %s -> dst node: %s", task[0], task[1]), false);
                    String SSSP_Dijkstra = getShortestPathsFromDijkstra(sqlContext, g, task[0], task[1]);
                    //check run time for task
                    run_time_report.addInfo(String.format("path found: %s", SSSP_Dijkstra), true);
                    all_results.add(SSSP_Dijkstra);
                    task_line = input_tasks.readLine();
                }
                break;
            case "DijkstraV2":
                /*Dijkstra V2*/
                sqlContext.cacheTable("relationDF");
                while(task_line != null) {
                    String[] task = task_line.split(" ");
                    run_time_report.addInfo(String.format("DijkstraV2 Processing task: \nsrc node: %s -> dst node: %s", task[0], task[1]), false);
                    String SSSP_DijkstraV2 = getShortestPathsFromDijkstraV2(sqlContext, task[0], task[1], noOfNodes);
                    //check run time for task
                    run_time_report.addInfo(String.format("path found: %s", SSSP_DijkstraV2), true);
                    all_results.add(SSSP_DijkstraV2);
                    task_line = input_tasks.readLine();
                }
                break;
            case "DijkstraV3":
                run_time_report.addInfo(String.format("DijkstraV3 Processing all tasks:"), false);
                Dataset<Row> taskDF = sqlContext.read()
                        .text(input_src_dst_file)
                        .withColumn("temp", split(col("value"), " "))
                        .select(col("value").as("taskId"),
                                col("temp").getItem(0).as("from_node"),
                                col("temp").getItem(1).as("to_node"))
                        .drop("temp");
                taskDF.createOrReplaceTempView("taskDF");
                all_results = getShortestPathsFromDijkstraV3(sqlContext, noOfNodes, run_time_report);
                run_time_report.addInfo(String.format("path found: \n%s", String.join( "\n", all_results)), true);
                break;
            default:
                fs.close();
                jsc.stop();
                System.out.println("no algorithm found");
                System.exit(0);
        }


        sqlContext.createDataset(all_results, Encoders.STRING()).coalesce(1).write().text( "output2");
        file = fs.globStatus(new Path(String.format("%s/part*", "output2")))[0].getPath().getName();
        System.out.println(file);
        fs.rename(new Path(String.format("%s/%s", "output2",file)), new Path(result_path_file));
        fs.delete(new Path("output2"), true);

        fs.close();
        jsc.stop();

        run_time_report.addInfo("Job done overall", true);
        run_time_report.printReport();

    }
}