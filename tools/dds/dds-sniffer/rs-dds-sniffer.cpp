// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2022 Intel Corporation. All Rights Reserved.

#include "rs-dds-sniffer.h"

#include <thread>
#include <memory>

#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/DataReaderListener.hpp>
#include <fastdds/dds/log/Log.hpp>
#include <fastrtps/types/DynamicDataHelper.hpp>
#include <fastrtps/types/DynamicDataFactory.h>

#include <tclap/CmdLine.h>
#include <tclap/ValueArg.h>
#include <tclap/SwitchArg.h>

#include <librealsense2/utilities/easylogging/easyloggingpp.h>
#include <librealsense2/rs.hpp>  // Include RealSense Cross Platform API
#include <realdds/dds-utilities.h>
#include <realdds/dds-guid.h>

using namespace TCLAP;
using namespace eprosima::fastdds::dds;
using namespace eprosima::fastrtps::types;
using librealsense::dds::print;

// FastDDS GUID_t: (MSB first, little-endian; see GuidUtils.hpp)
//     2 bytes  -  vendor ID
//     2 bytes  -  host
//     4 bytes  -  process (2 pid, 2 random)
//     4 bytes  -  participant
//     4 bytes  -  entity ID (reader/writer)
// For example:
//  Participant 1                 - 01.0f.be.05.f0.09.86.b6.01.00.00.00|0.0.1.c1
//  Writer under participant 1    - 01.0f.be.05.f0.09.86.b6.01.00.00.00|0.0.1.2
//  Participant 2 of same process - 01.0f.be.05.f0.09.86.b6.02.00.00.00|0.0.1.c1
//  Reader under participant 2    - 01.0f.be.05.f0.09.86.b6.02.00.00.00|0.0.1.7
//  Participant 3 other process   - 01.0f.be.05.88.50.ea.4a.01.00.00.00|0.0.1.c1
// Note same host for all, participant and entity IDs may be repeat for different processes
// To differentiate entities of different participant with same name we append process GUID values to the name
constexpr uint8_t GUID_PROCESS_LOCATION = 4;

// Redirect DDS log messages to our own logging mechanism
struct log_consumer : eprosima::fastdds::dds::LogConsumer
{
    virtual void Consume( const eprosima::fastdds::dds::Log::Entry & e ) override
    {
        using eprosima::fastdds::dds::Log;
        switch( e.kind )
        {
        case Log::Kind::Error:
            LOG_ERROR( "[DDS] " << e.message );
            break;
        case Log::Kind::Warning:
            LOG_WARNING( "[DDS] " << e.message );
            break;
        case Log::Kind::Info:
            LOG_DEBUG( "[DDS] " << e.message );
            break;
        }
    }
};


static eprosima::fastrtps::rtps::GuidPrefix_t std_prefix;


int main( int argc, char ** argv ) try
{
    librealsense::dds::dds_domain_id domain = 0;
    uint32_t seconds = 0;

    CmdLine cmd( "librealsense rs-dds-sniffer tool", ' ' );
    SwitchArg snapshot_arg( "s", "snapshot", "run momentarily taking a snapshot of the domain" );
    SwitchArg machine_readable_arg( "m", "machine-readable", "output entities in a way more suitable for automatic parsing" );
    SwitchArg topic_samples_arg( "t", "topic-samples", "register to topics that send TypeObject and print their samples" );
    SwitchArg debug_arg( "", "debug", "Enable debug logging", false );
    ValueArg< librealsense::dds::dds_domain_id > domain_arg( "d", "domain", "select domain ID to listen on", false, 0, "0-232" );
    cmd.add( snapshot_arg );
    cmd.add( machine_readable_arg );
    cmd.add( topic_samples_arg );
    cmd.add( domain_arg );
    cmd.add( debug_arg );
    cmd.parse( argc, argv );

    // Intercept DDS messages and redirect them to our own logging mechanism
    std::unique_ptr< eprosima::fastdds::dds::LogConsumer > consumer( new log_consumer() );
    eprosima::fastdds::dds::Log::ClearConsumers();
    eprosima::fastdds::dds::Log::RegisterConsumer( std::move( consumer ) );

#ifdef BUILD_SHARED_LIBS
    // Configure the same logger as librealsense, and default to only errors by default...
    el::Configurations defaultConf;
    defaultConf.setToDefault();
    defaultConf.setGlobally( el::ConfigurationType::ToStandardOutput, "false" );
    defaultConf.set( el::Level::Error, el::ConfigurationType::ToStandardOutput, "true" );
    defaultConf.setGlobally( el::ConfigurationType::Format, "-%levshort- %datetime{%H:%m:%s.%g} %msg (%fbase:%line [%thread])" );
    el::Loggers::reconfigureLogger( "librealsense", defaultConf );
#endif

    if( debug_arg.isSet() )
    {
        rs2::log_to_console( RS2_LOG_SEVERITY_DEBUG );
        eprosima::fastdds::dds::Log::SetVerbosity( eprosima::fastdds::dds::Log::Info );
    }
    else
    {
        rs2::log_to_console( RS2_LOG_SEVERITY_ERROR );
    }

    if( snapshot_arg.isSet() )
    {
        seconds = 3;
    }

    if( domain_arg.isSet() )
    {
        domain = domain_arg.getValue();
        if( domain > 232 )
        {
            LOG_ERROR( "Invalid domain value, enter a value in the range [0, 232]" );
            return EXIT_FAILURE;
        }
    }

    dds_sniffer snif;
    if( snif.init( domain, snapshot_arg.isSet(), machine_readable_arg.isSet(), topic_samples_arg.isSet() ) )
    {
        std_prefix = snif.get_participant().guid().guidPrefix;
        snif.run( seconds );
    }
    else
    {
        LOG_ERROR( "Initialization failure" );
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
catch( const TCLAP::ExitException & )
{
    LOG_ERROR( "Undefined exception while parsing command line arguments" );
    return EXIT_FAILURE;
}
catch( const TCLAP::ArgException & e )
{
    LOG_ERROR( e.what() );
    return EXIT_FAILURE;
}
catch( const std::exception & e )
{
    LOG_ERROR( e.what() );
    return EXIT_FAILURE;
}

dds_sniffer::dds_sniffer()
    : _participant()
    , _reader_listener( _discovered_types_datas )
{
}

dds_sniffer::~dds_sniffer()
{
    for( const auto & it : _discovered_types_readers )
    {
        DDS_API_CALL_NO_THROW( _discovered_types_subscriber->delete_datareader( it.first ) );  // If not empty than _discovered_types_subscriber != nullptr
        DDS_API_CALL_NO_THROW( _participant.get()->delete_topic( it.second ) );
    }

    if( _discovered_types_subscriber != nullptr )
    {
        DDS_API_CALL_NO_THROW( _participant.get()->delete_subscriber( _discovered_types_subscriber ) );
    }

    _discovered_types_readers.clear();
    _discovered_types_datas.clear();
}

bool dds_sniffer::init( librealsense::dds::dds_domain_id domain,
                        bool snapshot,
                        bool machine_readable,
                        bool topic_samples )
{
    _print_discoveries = ! snapshot;
    _print_by_topics = snapshot;
    _print_machine_readable = machine_readable;
    _print_topic_samples = topic_samples && ! snapshot;

    // Set callbacks before calling _participant.init(), or some events, specifically on_participant_added, might get lost
    _participant.create_listener( &_listener )
        ->on_writer_added( [this]( librealsense::dds::dds_guid guid, char const * topic_name ) {
            on_writer_added( guid, topic_name );
        } )
        ->on_writer_removed( [this]( librealsense::dds::dds_guid guid, char const * topic_name ) {
            on_writer_removed( guid, topic_name );
        } )
        ->on_reader_added( [this]( librealsense::dds::dds_guid guid, char const * topic_name ) {
            on_reader_added( guid, topic_name );
        } )
        ->on_reader_removed( [this]( librealsense::dds::dds_guid guid, char const * topic_name ) {
            on_reader_removed( guid, topic_name );
        } )
        ->on_participant_added( [this]( librealsense::dds::dds_guid guid, char const * participant_name ) {
            on_participant_added( guid, participant_name );
        } )
        ->on_participant_removed( [this]( librealsense::dds::dds_guid guid, char const * participant_name ) {
            on_participant_removed( guid, participant_name );
        } )
        ->on_type_discovery( [this]( char const * topic_name, DynamicType_ptr dyn_type ) {
            on_type_discovery( topic_name, dyn_type );
        } );

    _participant.init( domain, "rs-dds-sniffer" );

    if( ! _print_machine_readable )
    {
        std::cout << "rs-dds-sniffer (" << print( _participant.guid() ) << ") ";
        if( snapshot )
            std::cout << "taking a snapshot of ";
        else
            std::cout << "listening on ";
        std::cout << "domain " << domain << std::endl;
    }

    return _participant.is_valid();
}

void dds_sniffer::run( uint32_t seconds )
{
    if( seconds == 0 )
    {
        std::cin.ignore( std::numeric_limits< std::streamsize >::max() );
    }
    else
    {
        std::this_thread::sleep_for( std::chrono::seconds( seconds ) );
    }

    if( _print_by_topics )
    {
        if( _print_machine_readable )
        {
            print_topics_machine_readable();
        }
        else
        {
            print_topics();
        }
    }
}

void dds_sniffer::on_writer_added( librealsense::dds::dds_guid guid, const char * topic_name )
{
    if( _print_discoveries )
    {
        print_writer_discovered( guid, topic_name, true );
    }

    save_topic_writer( guid, topic_name );
}

void dds_sniffer::on_writer_removed( librealsense::dds::dds_guid guid, const char * topic_name )
{
    if( _print_discoveries )
    {
        print_writer_discovered( guid, topic_name, false );
    }

    remove_topic_writer( guid, topic_name );
}

void dds_sniffer::on_reader_added( librealsense::dds::dds_guid guid, const char * topic_name )
{
    if( _print_discoveries )
    {
        print_reader_discovered( guid, topic_name, true );
    }
    save_topic_reader( guid, topic_name );
}

void dds_sniffer::on_reader_removed( librealsense::dds::dds_guid guid, const char * topic_name )
{
    if( _print_discoveries )
    {
        print_reader_discovered( guid, topic_name, false );
    }

    remove_topic_reader( guid, topic_name );
}

void dds_sniffer::on_participant_added( librealsense::dds::dds_guid guid, const char * participant_name )
{
    if( _print_discoveries )
    {
        print_participant_discovered( guid, participant_name, true );
    }

    std::lock_guard< std::mutex > lock( _dds_entities_lock );

    _discovered_participants[guid] = participant_name;
}

void dds_sniffer::on_participant_removed( librealsense::dds::dds_guid guid, const char * participant_name )
{
    if( _print_discoveries )
    {
        print_participant_discovered( guid, participant_name, false );
    }

    std::lock_guard< std::mutex > lock( _dds_entities_lock );

    _discovered_participants.erase( guid );
}

void dds_sniffer::on_type_discovery( char const * topic_name, DynamicType_ptr dyn_type )
{
    if( ! _print_by_topics )
    {
        // Register type with participant
        TypeSupport type_support( DDS_API_CALL( new DynamicPubSubType( dyn_type ) ) );
        DDS_API_CALL( type_support.register_type( _participant.get() ) );
        std::cout << "Discovered topic '" << topic_name << "' of type '" << type_support->getName() << "'" << std::endl;

        if( _print_topic_samples )
        {
            // Create subscriber, topic and reader to receive instances of this topic
            if( _discovered_types_subscriber == nullptr )
            {
                _discovered_types_subscriber = DDS_API_CALL( _participant.get()->create_subscriber( SUBSCRIBER_QOS_DEFAULT,
                                                                                                    nullptr ) );
                if( _discovered_types_subscriber == nullptr )
                {
                    LOG_ERROR( "Cannot create subscriber for discovered type '" << topic_name );
                    return;
                }
            }

            Topic * topic = DDS_API_CALL( _participant.get()->create_topic( topic_name, type_support->getName(),
                                                                            TOPIC_QOS_DEFAULT ) );
            if( topic == nullptr )
            {
                LOG_ERROR( "Cannot create topic for discovered type '" << topic_name );
                return;
            }

            StatusMask sub_mask = StatusMask::subscription_matched() << StatusMask::data_available();
            DataReader * reader = DDS_API_CALL( _discovered_types_subscriber->create_datareader( topic,
                                                                                                 DATAREADER_QOS_DEFAULT,
                                                                                                 &_reader_listener,
                                                                                                 sub_mask ) );
            if( reader == nullptr )
            {
                LOG_ERROR( "Cannot create reader for discovered type '" << topic_name );
                DDS_API_CALL( _participant.get()->delete_topic( topic ) );
                return;
            }
            _discovered_types_readers[reader] = topic;

            DynamicData_ptr data( DDS_API_CALL( DynamicDataFactory::get_instance()->create_data( dyn_type ) ) );
            _discovered_types_datas[reader] = data;
        }
    }
}

dds_sniffer::dds_reader_listener::dds_reader_listener( std::map< DataReader *, DynamicData_ptr > & datas )
    : _datas( datas )
{
}

void dds_sniffer::dds_reader_listener::on_data_available( DataReader * reader )
{
    const TopicDescription * topic_desc = DDS_API_CALL( reader->get_topicdescription() );
    std::cout << "Received topic " << topic_desc->get_name() << " of type " 
              << topic_desc->get_type_name() << std::endl;

    auto dit = _datas.find( reader );

    if( dit != _datas.end() )
    {
        DynamicData_ptr data = dit->second;
        SampleInfo info;
        if( DDS_API_CALL( reader->take_next_sample( data.get(), &info ) ) == ReturnCode_t::RETCODE_OK )
        {
            if( info.valid_data )
            {
                DynamicDataHelper::print( data );
            }
        }
    }
}

void dds_sniffer::dds_reader_listener::on_subscription_matched( DataReader *, const SubscriptionMatchedStatus & info )
{
    if( info.current_count_change == 1 )
    {
        LOG_DEBUG( "Subscriber matched" );
    }
    else if( info.current_count_change == -1 )
    {
        LOG_DEBUG( "Subscriber unmatched" );
    }
    else
    {
        LOG_ERROR( info.current_count_change << " is not a valid value for SubscriptionMatchedStatus current count change" );
    }
}

void dds_sniffer::save_topic_writer( librealsense::dds::dds_guid guid, const char * topic_name )
{
    std::lock_guard< std::mutex > lock( _dds_entities_lock );

    _topics_info_by_name[topic_name].writers.insert( guid );
}

void dds_sniffer::remove_topic_writer( librealsense::dds::dds_guid guid, const char * topic_name )
{
    std::lock_guard< std::mutex > lock( _dds_entities_lock );

    auto topic_entry = _topics_info_by_name.find( topic_name );
    if( topic_entry != _topics_info_by_name.end() )
    {
        topic_entry->second.writers.erase( guid );
        if( topic_entry->second.writers.empty() && topic_entry->second.readers.empty() )
        {
            _topics_info_by_name.erase( topic_entry );
        }
    }
}

void dds_sniffer::save_topic_reader( librealsense::dds::dds_guid guid, const char * topic_name )
{
    std::lock_guard< std::mutex > lock( _dds_entities_lock );

    _topics_info_by_name[topic_name].readers.insert( guid );
}

void dds_sniffer::remove_topic_reader( librealsense::dds::dds_guid guid, const char * topic_name )
{
    std::lock_guard< std::mutex > lock( _dds_entities_lock );

    auto topic_entry = _topics_info_by_name.find( topic_name );
    if( topic_entry != _topics_info_by_name.end() )
    {
        topic_entry->second.readers.erase( guid );
        if( topic_entry->second.writers.empty() && topic_entry->second.readers.empty() )
        {
            _topics_info_by_name.erase( topic_entry );
        }
    }
}

uint32_t dds_sniffer::calc_max_indentation() const
{
    uint32_t indentation = 0;
    uint32_t max_indentation = 0;

    for( auto topic : _topics_info_by_name )  //_dds_entities_lock locked by print_topics()
    {
        // Use / as delimiter for nested topic names
        indentation = static_cast< uint32_t >( std::count( topic.first.begin(), topic.first.end(), '/' ) );
        if( indentation >= max_indentation )
        {
            max_indentation = indentation + 1;  //+1 for Reader/Writer indentation
        }
    }

    return max_indentation;
}

void dds_sniffer::print_writer_discovered( librealsense::dds::dds_guid guid,
                                           const char * topic_name,
                                           bool discovered ) const
{
    if( _print_machine_readable )
    {
        std::cout << "DataWriter," << print( guid, std_prefix ) << "," << topic_name
                  << ( discovered ? ",discovered" : ",removed" ) << std::endl;
    }
    else
    {
        std::cout << "DataWriter " << print( guid, std_prefix ) << " publishing topic '" << topic_name
                  << ( discovered ? "' discovered" : "' removed" ) << std::endl;
    }
}

void dds_sniffer::print_reader_discovered( librealsense::dds::dds_guid guid,
                                           const char * topic_name,
                                           bool discovered ) const
{
    if( _print_machine_readable )
    {
        std::cout << "DataReader," << print( guid, std_prefix ) << "," << topic_name
                  << ( discovered ? ",discovered" : ",removed" ) << std::endl;
    }
    else
    {
        std::cout << "DataReader " << print( guid, std_prefix ) << " reading topic '" << topic_name
                  << ( discovered ? "' discovered" : "' removed" ) << std::endl;
    }
}

void dds_sniffer::print_participant_discovered( librealsense::dds::dds_guid guid,
                                                const char * participant_name,
                                                bool discovered ) const
{
    if( _print_machine_readable )
    {
        std::cout << "Participant," << print( guid, std_prefix ) << "," << participant_name
                  << ( discovered ? ",discovered" : ",removed" )
                  << std::endl;
    }
    else
    {
        //prefix_.value[4] = static_cast<octet>( pid & 0xFF );
        //prefix_.value[5] = static_cast<octet>( ( pid >> 8 ) & 0xFF );
        uint16_t pid
            = guid.guidPrefix.value[GUID_PROCESS_LOCATION] + ( guid.guidPrefix.value[GUID_PROCESS_LOCATION + 1] << 8 );
        std::cout << "Participant " << print( guid, std_prefix ) << " '" << participant_name << "' (" << std::hex << pid
                  << std::dec << ") " << ( discovered ? " discovered" : " removed" ) << std::endl;
    }
}

void dds_sniffer::print_topics_machine_readable() const
{
    std::lock_guard< std::mutex > lock( _dds_entities_lock );

    for( auto topic : _topics_info_by_name )
    {
        for( auto writer : topic.second.writers )
        {
            std::cout << topic.first << ",";
            print_topic_writer( writer );
        }
        for( auto reader : topic.second.readers )
        {
            std::cout << topic.first << ",";
            print_topic_reader( reader );
        }
    }
}

void dds_sniffer::print_topics() const
{
    std::istringstream last_topic( "" );
    std::string last_topic_nested;

    std::lock_guard< std::mutex > lock( _dds_entities_lock );

    uint32_t max_indentation( calc_max_indentation() );

    for( auto topic : _topics_info_by_name )
    {
        std::cout << std::endl;

        std::istringstream current_topic( topic.first );  // Get topic name
        std::string current_topic_nested;
        uint32_t indentation = 0;

        // Compare to previous topic
        while( std::getline( last_topic, last_topic_nested, '/' ) )  // Use / as delimiter for nested topic names
        {
            if( std::getline( current_topic, current_topic_nested, '/' ) )
            {
                if( current_topic_nested.compare( last_topic_nested ) == 0 )
                {
                    ++indentation;  // Skip parts that are same as previous topic
                }
                else
                {
                    ident( indentation );
                    std::cout << current_topic_nested << std::endl;
                    ++indentation;
                    break;
                }
            }
        }

        // Print reminder of string
        while( std::getline( current_topic, current_topic_nested, '/' ) )
        {
            ident( indentation );
            std::cout << current_topic_nested << std::endl;
            ++indentation;
        }

        for( auto writer : topic.second.writers )
        {
            print_topic_writer( writer, max_indentation );
        }
        for( auto reader : topic.second.readers )
        {
            print_topic_reader( reader, max_indentation );
        }

        last_topic.clear();
        last_topic.str( topic.first );  // Save topic name for next iteration
        last_topic.seekg( 0, last_topic.beg );
    }
}

void dds_sniffer::ident( uint32_t indentation ) const
{
    while( indentation > 0 )
    {
        std::cout << "    ";
        --indentation;
    }
    std::cout << "- ";
}

void dds_sniffer::print_topic_writer( librealsense::dds::dds_guid guid, uint32_t indentation ) const
{
    auto iter = _discovered_participants.begin();
    for( ; iter != _discovered_participants.end(); ++iter )  //_dds_entities_lock locked by caller
    {
        if( iter->first.guidPrefix == guid.guidPrefix )
        {
            uint16_t tmp;
            memcpy( &tmp, &iter->first.guidPrefix.value[GUID_PROCESS_LOCATION], sizeof( tmp ) );
            if( _print_machine_readable )
            {
                std::cout << "Writer," << iter->second << "_" << std::hex << std::setw( 4 ) << std::setfill( '0' )
                          << tmp << std::dec << std::endl;
            }
            else
            {
                ident( indentation );
                std::cout << "Writer of \"" << iter->second << "_" << std::hex << std::setw( 4 ) << std::setfill( '0' )
                          << tmp << std::dec << "\"" << std::endl;
            }
            break;
        }
    }
    if( iter == _discovered_participants.end() )
    {
        ident( indentation );
        std::cout << "Writer of unknown participant" << std::endl;
    }
}

void dds_sniffer::print_topic_reader( librealsense::dds::dds_guid guid, uint32_t indentation ) const
{
    auto iter = _discovered_participants.begin();
    for( ; iter != _discovered_participants.end(); ++iter )  //_dds_entities_lock locked by caller
    {
        if( iter->first.guidPrefix == guid.guidPrefix )
        {
            uint16_t tmp;
            memcpy( &tmp, &iter->first.guidPrefix.value[GUID_PROCESS_LOCATION], sizeof( tmp ) );
            if( _print_machine_readable )
            {
                std::cout << "Reader," << iter->second << "_" << std::hex << std::setw( 4 ) << std::setfill( '0' )
                          << tmp << std::dec << std::endl;
            }
            else
            {
                ident( indentation );
                std::cout << "Reader of \"" << iter->second << "_" << std::hex << std::setw( 4 ) << std::setfill( '0' )
                          << tmp << std::dec << "\"" << std::endl;
            }
            break;
        }
    }
    if( iter == _discovered_participants.end() )
    {
        ident( indentation );
        std::cout << "Reader of unknown participant" << std::endl;
    }
}
